import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

from opendock.core.clustering import BaseCluster
from opendock.core.conformation import LigandConformation, ReceptorConformation
from opendock.core.io import write_ligand_traj
from opendock.sampler.bayesian import BayesianOptimizationSampler
from opendock.sampler.ga import GeneticAlgorithmSampler
from opendock.sampler.minimizer import adam_minimizer, lbfgs_minimizer, sgd_minimizer
from opendock.sampler.monte_carlo import MonteCarloSampler
from opendock.sampler.particle_swarm import ParticleSwarmOptimizer
from opendock.scorer.deeprmsd import DRmsdVinaSF, DeepRmsdSF
from opendock.scorer.hybrid import HybridSF
from opendock.scorer.onionnet_sfct import OnionNetSFCTSF
from opendock.scorer.scoring_function import BaseScoringFunction
from opendock.scorer.vina import VinaSF
from opendock.scorer.zPoseRanker import zPoseRankerSF

SAMPLERS = {
    "mc": MonteCarloSampler,
    "ga": GeneticAlgorithmSampler,
    "pso": ParticleSwarmOptimizer,
    "bo": BayesianOptimizationSampler,
}

SAMPLER_STEP_FACTOR = {
    "mc": 100,
    "ga": 10,
    "pso": 10,
    "bo": 20,
}

PRIMARY_SCORERS = {
    "vina": VinaSF,
    "deeprmsd": DeepRmsdSF,
    "rmsd-vina": DRmsdVinaSF,
}

RESCORERS = {
    "vina": VinaSF,
    "deeprmsd": DeepRmsdSF,
    "rmsd-vina": DRmsdVinaSF,
    "zranker": zPoseRankerSF,
    "sfct": OnionNetSFCTSF,
}

MINIMIZERS = {
    "adam": adam_minimizer,
    "lbfgs": lbfgs_minimizer,
    "sgd": sgd_minimizer,
    "none": None,
}


def _enable_deeprmsd_torch_load_compat() -> None:
    """Compat for legacy DeepRMSD pickle on newer PyTorch.

    OpenDock's DeepRMSD model file is a pickled module object. PyTorch 2.6+
    defaults torch.load(..., weights_only=True), which rejects this file.
    We retry with weights_only=False only for the DeepRMSD model checkpoint.
    """
    if getattr(torch, "_opendocker_deeprmsd_torchload_patched", False):
        return

    # Legacy DeepRMSD checkpoints may reference __main__.CNN.
    try:
        from opendock.scorer.deeprmsd import CNN as _DeepRMSDCNN

        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "CNN"):
            setattr(main_mod, "CNN", _DeepRMSDCNN)

        try:
            torch.serialization.add_safe_globals([_DeepRMSDCNN])
        except Exception:
            pass
    except Exception:
        _DeepRMSDCNN = None

    original_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        try:
            return original_load(f, *args, **kwargs)
        except Exception as exc:
            path = str(f)
            msg = str(exc)
            if "deeprmsd_model" in path and (
                "Weights only load failed" in msg or "__main__.CNN" in msg or "Can't get attribute 'CNN'" in msg
            ):
                retry_kwargs = dict(kwargs)
                retry_kwargs["weights_only"] = False
                return original_load(f, *args, **retry_kwargs)
            raise

    torch.load = _patched_torch_load
    torch._opendocker_deeprmsd_torchload_patched = True


@dataclass
class LigandInput:
    ligand_id: str
    smiles: str


class AnchorDistanceRestraintSF(BaseScoringFunction):
    """Pairwise ligand-core to receptor-anchor flat-bottom restraints.

    For each pair i, with distance d_i and reference distance r_i,
    penalty_i = force * max(0, |d_i - r_i| - tolerance)^2
    Total score is mean(penalty_i) over all pairs.
    """

    def __init__(
        self,
        receptor=None,
        ligand=None,
        receptor_indices=None,
        ligand_indices=None,
        reference_distances=None,
        tolerance=0.5,
        force_constant=1.0,
    ):
        super().__init__(receptor=receptor, ligand=ligand)
        self.receptor_indices = list(receptor_indices or [])
        self.ligand_indices = list(ligand_indices or [])
        self.reference_distances = torch.tensor(reference_distances or [], dtype=torch.float32)
        self.tolerance = float(tolerance)
        self.force_constant = float(force_constant)

        if len(self.receptor_indices) != len(self.ligand_indices):
            raise ValueError("AnchorDistanceRestraintSF index length mismatch")
        if len(self.receptor_indices) != int(self.reference_distances.numel()):
            raise ValueError("AnchorDistanceRestraintSF distance length mismatch")

    def scoring(self):
        nposes = len(self.ligand.pose_heavy_atoms_coords)
        if not self.receptor_indices:
            return torch.zeros((nposes, 1), dtype=torch.float32)

        lig_xyz = self.ligand.pose_heavy_atoms_coords[:, self.ligand_indices, :]
        rec_xyz = self.receptor.rec_heavy_atoms_xyz[self.receptor_indices, :].unsqueeze(0)
        rec_xyz = rec_xyz.expand(lig_xyz.size(0), -1, -1)

        d = torch.sqrt(torch.sum((lig_xyz - rec_xyz) ** 2, dim=2))
        ref = self.reference_distances.unsqueeze(0).expand_as(d)

        lower = ref - self.tolerance
        upper = ref + self.tolerance

        low_violation = torch.relu(lower - d)
        high_violation = torch.relu(d - upper)

        penalty = self.force_constant * (low_violation**2 + high_violation**2)
        return torch.mean(penalty, dim=1).reshape((-1, 1))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _must_get(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"Missing required key: {key}")
    return d[key]


def _validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    inputs = _must_get(cfg, "inputs")
    box = _must_get(cfg, "box")
    docking = _must_get(cfg, "docking")
    optimization = _must_get(cfg, "optimization")
    rescoring = _must_get(cfg, "rescoring")
    constraints = _must_get(cfg, "constraints")
    runtime = _must_get(cfg, "runtime")
    outputs = _must_get(cfg, "outputs")

    if "receptor_pdbqt" not in inputs and "receptor_pdb" not in inputs:
        raise ValueError("inputs must include either receptor_pdbqt or receptor_pdb")
    for k in ["ligands_smiles", "reference_core_sdf", "reference_core_smarts"]:
        _must_get(inputs, k)

    if box.get("mode") != "from_reference_ligand":
        raise ValueError("box.mode must be 'from_reference_ligand' in v1")
    if len(_must_get(box, "size")) != 3:
        raise ValueError("box.size must have 3 elements")

    primary = _must_get(docking, "primary_scorer")
    docking.setdefault("enabled", True)
    if primary not in PRIMARY_SCORERS:
        raise ValueError(f"Unsupported docking.primary_scorer: {primary}")
    if bool(docking.get("enabled", True)):
        sampler = _must_get(docking, "sampler")
        if sampler not in SAMPLERS:
            raise ValueError(f"Unsupported docking.sampler: {sampler}")
    else:
        docking.setdefault("sampler", "mc")

    min_name = _must_get(optimization, "minimizer")
    if min_name not in MINIMIZERS:
        raise ValueError(f"Unsupported optimization.minimizer: {min_name}")
    if min_name == "none":
        raise ValueError(
            "optimization.minimizer='none' is not supported by this OpenDock build for sampling; use one of: lbfgs, adam, sgd"
        )
    opt_scorer = optimization.get("scorer")
    if opt_scorer is not None and opt_scorer not in PRIMARY_SCORERS:
        raise ValueError(f"Unsupported optimization.scorer: {opt_scorer}")
    if "freeze_rigid_body" not in optimization:
        optimization["freeze_rigid_body"] = not bool(docking.get("enabled", True))

    rescore_name = _must_get(rescoring, "scorer")
    if rescoring.get("enabled", False) and rescore_name not in RESCORERS:
        raise ValueError(f"Unsupported rescoring.scorer: {rescore_name}")

    core = _must_get(constraints, "core")
    if core.get("enabled", False):
        if float(core.get("weight", 0.0)) < 0:
            raise ValueError("constraints.core.weight must be >= 0")
        core.setdefault("tolerance", 0.5)
        core.setdefault("force_constant", 1.0)
        core.setdefault("max_query_mappings", 16)
        core.setdefault("template_constrained_embed", True)
        core.setdefault("template_constrained_embed_max_attempts", 20)
        core.setdefault("rigid_prealign", True)
        core.setdefault("init_conformers", {})
        init_cf = core["init_conformers"]
        if not isinstance(init_cf, dict):
            raise ValueError("constraints.core.init_conformers must be a mapping")
        init_cf.setdefault("enabled", False)
        init_cf.setdefault("n_confs", 50)
        init_cf.setdefault("rmsd_threshold", 0.5)
        init_cf.setdefault("max_output_conformers", None)
        init_cf.setdefault("embed_max_attempts", 0)
        init_cf.setdefault("embed_seed", 1234)
        if float(core["tolerance"]) < 0:
            raise ValueError("constraints.core.tolerance must be >= 0")
        if float(core["force_constant"]) < 0:
            raise ValueError("constraints.core.force_constant must be >= 0")
        if int(core["max_query_mappings"]) < 1:
            raise ValueError("constraints.core.max_query_mappings must be >= 1")
        if int(core["template_constrained_embed_max_attempts"]) < 1:
            raise ValueError("constraints.core.template_constrained_embed_max_attempts must be >= 1")
        if int(init_cf["n_confs"]) < 1:
            raise ValueError("constraints.core.init_conformers.n_confs must be >= 1")
        if float(init_cf["rmsd_threshold"]) < 0:
            raise ValueError("constraints.core.init_conformers.rmsd_threshold must be >= 0")
        if int(init_cf["embed_max_attempts"]) < 0:
            raise ValueError("constraints.core.init_conformers.embed_max_attempts must be >= 0")
        if init_cf["max_output_conformers"] is not None and int(init_cf["max_output_conformers"]) < 1:
            raise ValueError("constraints.core.init_conformers.max_output_conformers must be >= 1")

    _must_get(runtime, "exhaustiveness")
    _must_get(runtime, "num_modes")
    runtime.setdefault("reuse_enabled", True)
    _must_get(outputs, "dir")
    _must_get(outputs, "top_n")
    outputs.setdefault("final_rmsd_filter", {})
    frf = outputs["final_rmsd_filter"]
    if not isinstance(frf, dict):
        raise ValueError("outputs.final_rmsd_filter must be a mapping")
    frf.setdefault("enabled", False)
    frf.setdefault("threshold", 0.5)
    if float(frf["threshold"]) < 0:
        raise ValueError("outputs.final_rmsd_filter.threshold must be >= 0")

    return cfg


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    return _validate_config(cfg)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_torch_save(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_yaml_dump(data: dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    os.replace(tmp, path)


def _safe_yaml_load(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            val = yaml.safe_load(fh)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def _safe_torch_load(path: str) -> Any | None:
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _read_smiles(path: str) -> list[LigandInput]:
    ligands: list[LigandInput] = []
    with open(path, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            smiles = parts[0]
            ligand_id = parts[1] if len(parts) > 1 else f"lig_{i+1:04d}"
            ligands.append(LigandInput(ligand_id=ligand_id, smiles=smiles))
    if not ligands:
        raise ValueError("No ligands found in inputs.ligands_smiles")
    return ligands


def _sanitize_id(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in x)


def _to_pdbqt(input_path: str, output_path: str) -> None:
    _run(["obabel", input_path, "-O", output_path])


def _prepare_receptor(receptor_path: str, work_dir: str) -> str:
    if receptor_path.lower().endswith(".pdbqt"):
        return os.path.abspath(receptor_path)
    rec_out = os.path.join(work_dir, "receptor.pdbqt")
    _to_pdbqt(receptor_path, rec_out)
    return rec_out


def _build_rdkit_3d_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0DE
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ValueError(f"RDKit embedding failed for SMILES: {smiles}")
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    return mol


def _write_rdkit_sdf(mol: Chem.Mol, out_path: str) -> None:
    writer = Chem.SDWriter(out_path)
    writer.write(mol)
    writer.close()


def _prepare_ligand_files(ligand: LigandInput, work_dir: str) -> tuple[str, str, Chem.Mol]:
    safe_id = _sanitize_id(ligand.ligand_id)
    mol = _build_rdkit_3d_from_smiles(ligand.smiles)
    sdf_path = os.path.join(work_dir, f"{safe_id}.input.sdf")
    pdbqt_path = os.path.join(work_dir, f"{safe_id}.input.pdbqt")
    _write_rdkit_sdf(mol, sdf_path)
    _to_pdbqt(sdf_path, pdbqt_path)
    return sdf_path, pdbqt_path, mol


def _rigid_align_rdkit_to_template_core(
    mol: Chem.Mol,
    ligand_core_match: tuple[int, ...],
    template_core_xyz: np.ndarray,
) -> Chem.Mol:
    if len(ligand_core_match) != int(template_core_xyz.shape[0]):
        raise ValueError("Rigid prealignment failed: core atom count mismatch")
    aligned = Chem.Mol(mol)
    conf = aligned.GetConformer()
    probe_xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in ligand_core_match],
        dtype=np.float64,
    )
    target_xyz = template_core_xyz.astype(np.float64)

    probe_center = probe_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)
    probe0 = probe_xyz - probe_center
    target0 = target_xyz - target_center

    h = probe0.T @ target0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_center - (probe_center @ r.T)

    for atom_idx in range(aligned.GetNumAtoms()):
        p = conf.GetAtomPosition(atom_idx)
        v = np.array([p.x, p.y, p.z], dtype=np.float64)
        v2 = v @ r.T + t
        conf.SetAtomPosition(atom_idx, (float(v2[0]), float(v2[1]), float(v2[2])))
    return aligned


def _constrained_embed_rdkit_to_template_core(
    mol: Chem.Mol,
    ligand_core_match: tuple[int, ...],
    template_core_xyz: np.ndarray,
    max_attempts: int,
) -> tuple[Chem.Mol, bool, str]:
    if len(ligand_core_match) != int(template_core_xyz.shape[0]):
        return mol, False, "constrained_embed_failed:core_atom_count_mismatch"

    # Mapping-aware constrained embedding using explicit coordMap.
    # This preserves alternative SMARTS mappings for symmetric cores.
    mapped = Chem.Mol(mol)
    coord_map = {}
    for i, atom_idx in enumerate(ligand_core_match):
        xyz = template_core_xyz[i]
        coord_map[int(atom_idx)] = Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))

    status = AllChem.EmbedMolecule(
        mapped,
        maxAttempts=int(max_attempts),
        randomSeed=0xC0DE,
        clearConfs=True,
        coordMap=coord_map,
        useRandomCoords=False,
    )
    if status == 0:
        return mapped, True, ""

    # Fallback for environments where coordMap embedding may fail.
    # This may collapse symmetric mappings, but keeps the run robust.
    # Build a query-derived core submolecule and place its atoms at template-core coordinates.
    idx_map: dict[int, int] = {}
    rw = Chem.RWMol()
    for src_idx in ligand_core_match:
        idx_map[int(src_idx)] = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(int(src_idx))))
    for bond in mol.GetBonds():
        b = int(bond.GetBeginAtomIdx())
        e = int(bond.GetEndAtomIdx())
        if b in idx_map and e in idx_map:
            rw.AddBond(idx_map[b], idx_map[e], bond.GetBondType())
    core = rw.GetMol()
    core_conf = Chem.Conformer(len(ligand_core_match))
    for i in range(len(ligand_core_match)):
        xyz = template_core_xyz[i]
        core_conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    core.AddConformer(core_conf, assignId=True)
    try:
        embedded = AllChem.ConstrainedEmbed(
            Chem.Mol(mol),
            core,
            maxAttempts=int(max_attempts),
            randomseed=0xC0DE,
            useTethers=True,
        )
        return embedded, True, "constrained_embed_warning:coordMap_failed_used_ConstrainedEmbed"
    except Exception as exc:
        return mol, False, f"constrained_embed_failed:{exc}"


def _single_conformer_copy(mol: Chem.Mol, conf_id: int) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer(int(conf_id))
    new_conf = Chem.Conformer(conf)
    out.RemoveAllConformers()
    out.AddConformer(new_conf, assignId=True)
    return out


def _embed_multiple_with_coord_map(
    mol: Chem.Mol,
    coord_map: dict[int, Point3D],
    n_confs: int,
    seed: int,
    max_attempts: int,
) -> tuple[list[int], Chem.Mol]:
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    kwargs = {
        "numConfs": int(n_confs),
        "randomSeed": int(seed),
        "clearConfs": True,
        "coordMap": coord_map,
        "useRandomCoords": False,
    }
    if int(max_attempts) > 0:
        kwargs["maxAttempts"] = int(max_attempts)
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(work, **kwargs))
        if conf_ids:
            return conf_ids, work
    except Exception:
        pass

    # Robust fallback for RDKit builds with limited EmbedMultipleConfs kwargs support.
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    conf_ids = []
    for i in range(int(n_confs)):
        per_seed = int(seed) + i
        try:
            if int(max_attempts) > 0:
                status = AllChem.EmbedMolecule(
                    work,
                    maxAttempts=int(max_attempts),
                    randomSeed=per_seed,
                    clearConfs=False,
                    coordMap=coord_map,
                    useRandomCoords=False,
                )
            else:
                status = AllChem.EmbedMolecule(
                    work,
                    randomSeed=per_seed,
                    clearConfs=False,
                    coordMap=coord_map,
                    useRandomCoords=False,
                )
            if int(status) == 0:
                conf_ids.append(work.GetNumConformers() - 1)
        except Exception:
            continue
    return conf_ids, work


def _symmetry_aware_heavy_rms(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """RMSD on heavy atoms with symmetry handling, without mutating inputs."""
    prb = Chem.Mol(mol_a)
    ref = Chem.Mol(mol_b)
    try:
        prb = Chem.RemoveHs(prb)
    except Exception:
        pass
    try:
        ref = Chem.RemoveHs(ref)
    except Exception:
        pass
    # GetBestRMS aligns probe internally, so always use fresh copies.
    return float(rdMolAlign.GetBestRMS(Chem.Mol(prb), Chem.Mol(ref), prbId=0, refId=0))


def _greedy_rmsd_filter(mol: Chem.Mol, conf_ids: list[int], rmsd_threshold: float, max_keep: int | None) -> list[int]:
    kept: list[int] = []
    kept_mols: list[Chem.Mol] = []
    for cid in conf_ids:
        candidate = _single_conformer_copy(mol, int(cid))
        if not kept:
            kept.append(int(cid))
            kept_mols.append(candidate)
        else:
            keep_this = True
            for kept_mol in kept_mols:
                try:
                    rms = _symmetry_aware_heavy_rms(candidate, kept_mol)
                except Exception:
                    rms = 0.0
                if rms < float(rmsd_threshold):
                    keep_this = False
                    break
            if keep_this:
                kept.append(int(cid))
                kept_mols.append(candidate)
        if max_keep is not None and len(kept) >= int(max_keep):
            break
    return kept


def _merge_unique_candidates_by_rmsd(
    accepted: list[dict[str, Any]],
    new_candidates: list[dict[str, Any]],
    rmsd_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    merged = list(accepted)
    removed = 0
    for cand in new_candidates:
        duplicate = False
        for kept in merged:
            try:
                rms = _symmetry_aware_heavy_rms(cand["mol"], kept["mol"])
            except Exception:
                rms = 0.0
            if rms < float(rmsd_threshold):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            merged.append(cand)
    return merged, removed


def _generate_mapping_init_conformers(
    map_lig_rdkit: Chem.Mol,
    ligand_core_match: tuple[int, ...],
    template_core_xyz: np.ndarray,
    conf_cfg: dict[str, Any],
) -> tuple[list[Chem.Mol], dict[str, Any], str]:
    if not bool(conf_cfg.get("enabled", False)):
        return [Chem.Mol(map_lig_rdkit)], {"generated_count": 1, "kept_count": 1}, ""

    coord_map: dict[int, Point3D] = {}
    for i, atom_idx in enumerate(ligand_core_match):
        xyz = template_core_xyz[i]
        coord_map[int(atom_idx)] = Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))

    conf_ids, work = _embed_multiple_with_coord_map(
        map_lig_rdkit,
        coord_map,
        int(conf_cfg.get("n_confs", 50)),
        int(conf_cfg.get("embed_seed", 1234)),
        int(conf_cfg.get("embed_max_attempts", 0)),
    )
    if not conf_ids:
        return [Chem.Mol(map_lig_rdkit)], {"generated_count": 0, "kept_count": 1}, "init_conformers_failed:no_embedded_confs"

    max_keep_val = conf_cfg.get("max_output_conformers")
    max_keep = int(max_keep_val) if max_keep_val is not None else None
    kept_ids = _greedy_rmsd_filter(
        work,
        conf_ids,
        float(conf_cfg.get("rmsd_threshold", 0.5)),
        max_keep,
    )
    if not kept_ids:
        return [Chem.Mol(map_lig_rdkit)], {"generated_count": len(conf_ids), "kept_count": 1}, "init_conformers_failed:all_filtered_by_rmsd"

    out_mols = [_single_conformer_copy(work, cid) for cid in kept_ids]
    info = {
        "generated_count": int(len(conf_ids)),
        "kept_count": int(len(out_mols)),
        "rmsd_threshold": float(conf_cfg.get("rmsd_threshold", 0.5)),
        "max_output_conformers": max_keep_val,
    }
    return out_mols, info, ""


def _load_reference_template(reference_sdf: str, reference_smarts: str) -> tuple[Chem.Mol, tuple[int, ...], np.ndarray]:
    suppl = Chem.SDMolSupplier(reference_sdf, removeHs=False)
    ref_mol = next((m for m in suppl if m is not None), None)
    if ref_mol is None:
        raise ValueError(f"Cannot read reference SDF: {reference_sdf}")

    patt = Chem.MolFromSmarts(reference_smarts)
    if patt is None:
        raise ValueError("Invalid inputs.reference_core_smarts")

    matches = ref_mol.GetSubstructMatches(patt)
    if not matches:
        raise ValueError("No SMARTS match in reference_core_sdf")

    template_match = matches[0]
    conf = ref_mol.GetConformer()
    template_xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in template_match],
        dtype=np.float32,
    )
    return ref_mol, template_match, template_xyz


def _rdkit_heavy_index_list(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def _rdkit_heavy_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])
    return np.array(coords, dtype=np.float32)


def _map_rdkit_heavy_to_opendock(ligand_od: LigandConformation, lig_mol: Chem.Mol) -> dict[int, int]:
    rd_heavy_atom_indices = _rdkit_heavy_index_list(lig_mol)
    rd_coords = _rdkit_heavy_coords(lig_mol)
    od_coords = ligand_od.init_lig_heavy_atoms_xyz.detach().numpy().reshape((-1, 3))

    if rd_coords.shape[0] != od_coords.shape[0]:
        raise ValueError("RDKit/OpenDock heavy atom count mismatch")

    mapping: dict[int, int] = {}
    used_od = set()
    for rd_local_i, rd_atom_idx in enumerate(rd_heavy_atom_indices):
        d = np.linalg.norm(od_coords - rd_coords[rd_local_i], axis=1)
        order = np.argsort(d)
        pick = None
        for candidate in order:
            if int(candidate) not in used_od:
                pick = int(candidate)
                break
        if pick is None:
            raise ValueError("Failed heavy atom mapping")
        used_od.add(pick)
        mapping[int(rd_atom_idx)] = pick
    return mapping


def _build_template_receptor_anchors(receptor_pdbqt: str, template_core_xyz: np.ndarray) -> tuple[list[int], list[float]]:
    center = torch.tensor(np.mean(template_core_xyz, axis=0), dtype=torch.float32).reshape((1, 3))
    receptor = ReceptorConformation(receptor_pdbqt, center, init_lig_heavy_atoms_xyz=None)
    rec_xyz = receptor.rec_heavy_atoms_xyz.detach().numpy().reshape((-1, 3))

    rec_indices: list[int] = []
    ref_distances: list[float] = []
    for atom_xyz in template_core_xyz:
        d = np.linalg.norm(rec_xyz - atom_xyz, axis=1)
        idx = int(np.argmin(d))
        rec_indices.append(idx)
        ref_distances.append(float(d[idx]))
    return rec_indices, ref_distances


def _score_conformation(ligand: LigandConformation, receptor: ReceptorConformation, scorer: BaseScoringFunction, cnfr: torch.Tensor) -> float:
    ligand.cnfr2xyz([cnfr])
    if receptor.cnfrs_ is not None:
        receptor.cnfr2xyz(receptor.cnfrs_)
    value = scorer.scoring().detach().numpy().reshape(-1)[0]
    return float(value)


def _core_rmsd_for_pose(ligand: LigandConformation, cnfr: torch.Tensor, ligand_core_indices: list[int], ref_core_xyz: np.ndarray) -> float:
    ligand.cnfr2xyz([cnfr])
    pose_xyz = ligand.pose_heavy_atoms_coords[:, ligand_core_indices, :]
    ref_xyz = torch.tensor(ref_core_xyz, dtype=torch.float32).unsqueeze(0)
    rmsd = torch.sqrt(torch.mean(torch.sum((pose_xyz - ref_xyz) ** 2, dim=2), dim=1)).detach().numpy()[0]
    return float(rmsd)


def _pose_mol_from_cnfr(
    ligand: LigandConformation,
    lig_rdkit: Chem.Mol,
    rd_to_od: dict[int, int],
    cnfr: torch.Tensor,
) -> Chem.Mol:
    ligand.cnfr2xyz([cnfr])
    od_xyz = ligand.pose_heavy_atoms_coords[0].detach().numpy()
    out = Chem.Mol(lig_rdkit)
    conf = out.GetConformer()
    for rd_idx, od_idx in rd_to_od.items():
        p = od_xyz[int(od_idx)]
        conf.SetAtomPosition(int(rd_idx), (float(p[0]), float(p[1]), float(p[2])))
    return out


def _filter_ranked_records_by_global_rmsd(
    ranked_records: list[dict[str, Any]],
    rmsd_threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in ranked_records:
        duplicate = False
        for k in kept:
            try:
                rms = _symmetry_aware_heavy_rms(row["pose_mol"], k["pose_mol"])
            except Exception:
                rms = 0.0
            if rms < float(rmsd_threshold):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def _write_pose_records_sdf(
    pose_records: list[dict[str, Any]],
    out_sdf: str,
    primary_name: str,
) -> None:
    _ensure_dir(os.path.dirname(out_sdf))
    writer = Chem.SDWriter(out_sdf)
    for row in pose_records:
        mol = Chem.Mol(row["pose_mol"])
        try:
            if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
                mol = Chem.AddHs(Chem.RemoveHs(mol), addCoords=True)
        except Exception:
            pass
        mol.SetProp(f"primary_{primary_name}", str(float(row["primary"])))
        mol.SetProp("rescore", str(float(row["rescore"])))
        mol.SetProp("core_rmsd", str(float(row["core_rmsd"])))
        if "mapping_index" in row:
            mol.SetProp("mapping_index", str(int(row["mapping_index"])))
        if "conformer_index" in row:
            mol.SetProp("conformer_index", str(int(row["conformer_index"])))
        writer.write(mol)
    writer.close()


def _to_serializable_cfg_subset(cfg: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {k: cfg[k] for k in keys if k in cfg}


def _build_stage_keys(
    cfg: dict[str, Any],
    ligand_input: LigandInput,
    ligand_core_match: tuple[int, ...],
    mapping_index: int,
    conformer_index: int,
    file_hashes: dict[str, str],
) -> dict[str, str]:
    query_smarts = cfg["inputs"].get("ligand_core_smarts") or cfg["inputs"]["reference_core_smarts"]
    base = {
        "ligand_id": ligand_input.ligand_id,
        "ligand_smiles": ligand_input.smiles,
        "mapping_index": int(mapping_index),
        "conformer_index": int(conformer_index),
        "ligand_core_match": [int(x) for x in ligand_core_match],
        "reference_core_smarts": cfg["inputs"]["reference_core_smarts"],
        "ligand_core_smarts": query_smarts,
        "file_hashes": file_hashes,
    }
    docking_payload = {
        "base": base,
        "box": cfg["box"],
        "docking": cfg["docking"],
        "constraints": cfg["constraints"],
        "runtime": _to_serializable_cfg_subset(cfg["runtime"], ["exhaustiveness", "num_modes"]),
    }
    docking_key = _stable_hash(docking_payload)

    optimization_payload = {
        "docking_key": docking_key,
        "optimization": cfg["optimization"],
        "constraints": cfg["constraints"],
    }
    optimization_key = _stable_hash(optimization_payload)

    scoring_payload = {
        "optimization_key": optimization_key,
        "rescoring": cfg["rescoring"],
        "outputs": _to_serializable_cfg_subset(cfg["outputs"], ["top_n"]),
        "primary_scorer": cfg["docking"]["primary_scorer"],
    }
    scoring_key = _stable_hash(scoring_payload)
    return {
        "docking": docking_key,
        "optimization": optimization_key,
        "scoring": scoring_key,
    }


def _serialize_cnfrs(cnfrs: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.tensor(c.detach().cpu().numpy(), dtype=torch.float32) for c in cnfrs]


def _load_cached_cnfrs(stage_dir: str, stage: str, expected_key: str) -> list[torch.Tensor] | None:
    meta_path = os.path.join(stage_dir, f"{stage}.meta.yaml")
    data_path = os.path.join(stage_dir, f"{stage}.pt")
    meta = _safe_yaml_load(meta_path)
    if not meta or meta.get("stage_key") != expected_key:
        return None
    payload = _safe_torch_load(data_path)
    if not isinstance(payload, dict):
        return None
    vals = payload.get("cnfrs")
    if not isinstance(vals, list):
        return None
    out: list[torch.Tensor] = []
    for v in vals:
        if not isinstance(v, torch.Tensor):
            return None
        out.append(torch.tensor(v.detach().cpu().numpy(), dtype=torch.float32))
    return out


def _save_cached_cnfrs(stage_dir: str, stage: str, stage_key: str, cnfrs: list[torch.Tensor]) -> None:
    _ensure_dir(stage_dir)
    meta_path = os.path.join(stage_dir, f"{stage}.meta.yaml")
    data_path = os.path.join(stage_dir, f"{stage}.pt")
    _atomic_torch_save({"cnfrs": _serialize_cnfrs(cnfrs)}, data_path)
    _atomic_yaml_dump({"stage_key": stage_key, "count": len(cnfrs)}, meta_path)


def _load_cached_scores(stage_dir: str, expected_key: str) -> list[dict[str, float]] | None:
    meta_path = os.path.join(stage_dir, "scoring.meta.yaml")
    data_path = os.path.join(stage_dir, "scoring.yaml")
    meta = _safe_yaml_load(meta_path)
    if not meta or meta.get("stage_key") != expected_key:
        return None
    payload = _safe_yaml_load(data_path)
    if not payload:
        return None
    rows = payload.get("pose_records")
    if not isinstance(rows, list):
        return None
    parsed: list[dict[str, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        try:
            parsed.append(
                {
                    "primary": float(row["primary"]),
                    "rescore": float(row["rescore"]),
                    "core_rmsd": float(row["core_rmsd"]),
                }
            )
        except Exception:
            return None
    return parsed


def _save_cached_scores(stage_dir: str, stage_key: str, pose_records: list[dict[str, Any]]) -> None:
    _ensure_dir(stage_dir)
    meta_path = os.path.join(stage_dir, "scoring.meta.yaml")
    data_path = os.path.join(stage_dir, "scoring.yaml")
    serializable = []
    for row in pose_records:
        serializable.append(
            {
                "primary": float(row["primary"]),
                "rescore": float(row["rescore"]),
                "core_rmsd": float(row["core_rmsd"]),
            }
        )
    _atomic_yaml_dump({"pose_records": serializable}, data_path)
    _atomic_yaml_dump({"stage_key": stage_key, "count": len(serializable)}, meta_path)


def _write_stage_pose_snapshot(lig_pdbqt: str, cnfrs: list[torch.Tensor], out_sdf: str, box_center: list[float]) -> None:
    if not cnfrs:
        return
    _ensure_dir(os.path.dirname(out_sdf))
    lig = LigandConformation(lig_pdbqt)
    lig.ligand_center[0][0] = box_center[0]
    lig.ligand_center[0][1] = box_center[1]
    lig.ligand_center[0][2] = box_center[2]
    tmp_pdbqt = out_sdf.replace(".sdf", ".pdbqt")
    write_ligand_traj(cnfrs, lig, tmp_pdbqt, information={})
    _to_pdbqt(tmp_pdbqt, out_sdf)


def _post_optimize(
    cnfrs: list[torch.Tensor],
    ligand: LigandConformation,
    scoring_function: BaseScoringFunction,
    minimizer_name: str,
    steps: int,
    lr: float,
    freeze_rigid_body: bool = False,
) -> list[torch.Tensor]:
    minimizer = MINIMIZERS[minimizer_name]
    if minimizer is None:
        return cnfrs

    optimized = []
    for cnfr in cnfrs:
        cnfr_tensor = torch.tensor(cnfr.detach().numpy(), dtype=torch.float32)
        if freeze_rigid_body and cnfr_tensor.ndim == 2 and cnfr_tensor.shape[1] > 6:
            fixed_head = cnfr_tensor[:, :6].detach()
            x = [cnfr_tensor[:, 6:].detach().clone().requires_grad_()]

            def _target(v):
                merged = torch.cat([fixed_head, v[0]], dim=1)
                ligand.cnfr2xyz([merged])
                return torch.sum(scoring_function.scoring())

            new_tail = minimizer(x, _target, nsteps=int(steps), lr=float(lr))[0].detach()
            optimized.append(torch.cat([fixed_head, new_tail], dim=1).detach())
        else:
            x = [cnfr_tensor.requires_grad_()]

            def _target(v):
                ligand.cnfr2xyz(v)
                return torch.sum(scoring_function.scoring())

            new_x = minimizer(x, _target, nsteps=int(steps), lr=float(lr))
            optimized.append(new_x[0].detach())
    return optimized


def _rank_pose_records(records: list[dict[str, float]], use_rescore: bool) -> list[dict[str, float]]:
    key = "rescore" if use_rescore else "primary"
    return sorted(records, key=lambda x: x[key])


def _build_mapping_manifest(
    receptor: ReceptorConformation,
    template_core_match: tuple[int, ...],
    ligand_core_match: tuple[int, ...],
    ligand_core_od_indices: list[int],
    anchor_rec_indices: list[int],
    anchor_ref_distances: list[float],
    best_pose_distances: list[float],
    tolerance: float,
    force_constant: float,
    weight: float,
) -> dict[str, Any]:
    mapping_rows = []
    for i in range(len(anchor_rec_indices)):
        rec_idx = int(anchor_rec_indices[i])
        try:
            rec_row = receptor.dataframe_ha_.iloc[rec_idx]
            rec_meta = {
                "atomname": str(rec_row.get("atomname", "")),
                "resname": str(rec_row.get("resname", "")),
                "chain": str(rec_row.get("chain", "")),
                "resSeq": str(rec_row.get("resSeq", "")),
            }
        except Exception:
            rec_meta = {"atomname": "", "resname": "", "chain": "", "resSeq": ""}

        d_ref = float(anchor_ref_distances[i])
        d_pose = float(best_pose_distances[i])
        violation = max(0.0, abs(d_pose - d_ref) - float(tolerance))

        mapping_rows.append(
            {
                "core_position": int(i),
                "template_atom_idx_rdkit": int(template_core_match[i]),
                "query_atom_idx_rdkit": int(ligand_core_match[i]),
                "query_atom_idx_opendock": int(ligand_core_od_indices[i]),
                "protein_heavy_idx": rec_idx,
                "protein_atom": rec_meta,
                "reference_distance_A": d_ref,
                "best_pose_distance_A": d_pose,
                "distance_delta_A": d_pose - d_ref,
                "flat_bottom_violation_A": violation,
            }
        )

    return {
        "constraint_model": "template_core_to_nearest_protein_heavy_flat_bottom_quadratic",
        "parameters": {
            "tolerance_A": float(tolerance),
            "force_constant": float(force_constant),
            "weight": float(weight),
        },
        "mapping": mapping_rows,
    }


def _dock_with_mapping(
    cfg: dict[str, Any],
    receptor_pdbqt: str,
    lig_pdbqt: str,
    lig_rdkit: Chem.Mol,
    template_core_match: tuple[int, ...],
    ligand_core_match: tuple[int, ...],
    mapping_index: int,
    template_core_xyz: np.ndarray,
    anchor_rec_indices: list[int],
    anchor_ref_distances: list[float],
    cache_stage_dir: str,
    stage_keys: dict[str, str],
    reuse_enabled: bool,
) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
    box_center = np.mean(template_core_xyz, axis=0).tolist()
    box_size = [float(x) for x in cfg["box"]["size"]]

    ligand = LigandConformation(lig_pdbqt)
    ligand.ligand_center[0][0] = box_center[0]
    ligand.ligand_center[0][1] = box_center[1]
    ligand.ligand_center[0][2] = box_center[2]

    receptor = ReceptorConformation(
        receptor_pdbqt,
        torch.tensor(box_center, dtype=torch.float32).reshape((1, 3)),
        init_lig_heavy_atoms_xyz=ligand.init_lig_heavy_atoms_xyz,
    )

    rd_to_od = _map_rdkit_heavy_to_opendock(ligand, lig_rdkit)
    lig_core_od_indices: list[int] = []
    for idx in ligand_core_match:
        if idx not in rd_to_od:
            raise ValueError("Query SMARTS includes non-heavy atom; use heavy-atom SMARTS for core restraints")
        lig_core_od_indices.append(rd_to_od[idx])

    primary_name = cfg["docking"]["primary_scorer"]
    primary_scorer = PRIMARY_SCORERS[primary_name](receptor=receptor, ligand=ligand)

    core_cfg = cfg["constraints"]["core"]
    rest_scorer: AnchorDistanceRestraintSF | None = None
    scoring_for_sampling: BaseScoringFunction = primary_scorer
    if core_cfg.get("enabled", False):
        rest_scorer = AnchorDistanceRestraintSF(
            receptor=receptor,
            ligand=ligand,
            receptor_indices=anchor_rec_indices,
            ligand_indices=lig_core_od_indices,
            reference_distances=anchor_ref_distances,
            tolerance=float(core_cfg["tolerance"]),
            force_constant=float(core_cfg["force_constant"]),
        )
        scoring_for_sampling = HybridSF(
            receptor=receptor,
            ligand=ligand,
            scorers=[primary_scorer, rest_scorer],
            weights=[1.0, float(core_cfg.get("weight", 0.0))],
        )

    docking_enabled = bool(cfg["docking"].get("enabled", True))
    sampler_name = cfg["docking"]["sampler"]
    opt_cfg = cfg["optimization"]

    reuse_info = {
        "mapping_index": int(mapping_index),
        "reused_docking": False,
        "reused_optimization": False,
        "reused_scoring": False,
        "notes": [],
    }

    clustered_cnfrs = None
    if reuse_enabled:
        clustered_cnfrs = _load_cached_cnfrs(cache_stage_dir, "docking", stage_keys["docking"])
        if clustered_cnfrs:
            reuse_info["reused_docking"] = True
            reuse_info["notes"].append("reused:docking")
    if not clustered_cnfrs:
        if docking_enabled:
            sampler_cls = SAMPLERS[sampler_name]
            step_factor = SAMPLER_STEP_FACTOR[sampler_name]
            nsteps = int(cfg["runtime"]["exhaustiveness"]) * step_factor * int(ligand.number_of_heavy_atoms)
            sampler_minimizer = MINIMIZERS[opt_cfg["minimizer"]]
            sampler = sampler_cls(
                ligand,
                receptor,
                scoring_for_sampling,
                box_center=box_center,
                box_size=box_size,
                random_start=True,
                minimizer=sampler_minimizer,
            )

            init_lig_cnfrs = [torch.tensor(ligand.init_cnfrs.detach().numpy())]
            init_rec_cnfrs = receptor.init_cnfrs
            ligand.cnfrs_, receptor.cnfrs_ = sampler._random_move(init_lig_cnfrs, init_rec_cnfrs)

            if sampler_name == "ga":
                sampler.sampling(n_gen=max(2, nsteps // 100), verbose=False)
            elif sampler_name == "bo":
                sampler.sampling(n_iter=max(10, nsteps // 100), init_points=5)
            else:
                sampler.sampling(nsteps=nsteps)

            if not sampler.ligand_cnfrs_history_:
                raise RuntimeError("Sampling finished without collected conformations")

            cluster = BaseCluster(
                sampler.ligand_cnfrs_history_,
                None,
                sampler.ligand_scores_history_,
                ligand,
                1,
            )
            _, clustered_cnfrs, _ = cluster.clustering(num_modes=int(cfg["runtime"]["num_modes"]))
            if not clustered_cnfrs:
                raise RuntimeError("No clustered poses generated")
        else:
            clustered_cnfrs = [torch.tensor(ligand.init_cnfrs.detach().numpy(), dtype=torch.float32)]
            reuse_info["notes"].append("docking_disabled:using_prepared_input")
        _save_cached_cnfrs(cache_stage_dir, "docking", stage_keys["docking"], clustered_cnfrs)
        _write_stage_pose_snapshot(
            lig_pdbqt=lig_pdbqt,
            cnfrs=clustered_cnfrs,
            out_sdf=os.path.join(cache_stage_dir, "docking_poses.sdf"),
            box_center=box_center,
        )

    optimized_cnfrs = clustered_cnfrs
    if opt_cfg.get("enabled", False):
        cached_optimized = None
        if reuse_enabled:
            cached_optimized = _load_cached_cnfrs(cache_stage_dir, "optimization", stage_keys["optimization"])
            if cached_optimized and len(cached_optimized) == len(clustered_cnfrs):
                optimized_cnfrs = cached_optimized
                reuse_info["reused_optimization"] = True
                reuse_info["notes"].append("reused:optimization")
            else:
                cached_optimized = None

        if cached_optimized is None:
            opt_primary_name = opt_cfg.get("scorer") or primary_name
            opt_primary_scorer = PRIMARY_SCORERS[opt_primary_name](receptor=receptor, ligand=ligand)
            scoring_for_optimization: BaseScoringFunction = opt_primary_scorer
            if core_cfg.get("enabled", False) and rest_scorer is not None:
                scoring_for_optimization = HybridSF(
                    receptor=receptor,
                    ligand=ligand,
                    scorers=[opt_primary_scorer, rest_scorer],
                    weights=[1.0, float(core_cfg.get("weight", 0.0))],
                )
            optimized_cnfrs = _post_optimize(
                clustered_cnfrs,
                ligand,
                scoring_for_optimization,
                opt_cfg["minimizer"],
                int(opt_cfg["steps"]),
                float(opt_cfg["lr"]),
                bool(opt_cfg.get("freeze_rigid_body", False)),
            )
            _save_cached_cnfrs(cache_stage_dir, "optimization", stage_keys["optimization"], optimized_cnfrs)
            _write_stage_pose_snapshot(
                lig_pdbqt=lig_pdbqt,
                cnfrs=optimized_cnfrs,
                out_sdf=os.path.join(cache_stage_dir, "optimization_poses.sdf"),
                box_center=box_center,
            )
    else:
        _save_cached_cnfrs(cache_stage_dir, "optimization", stage_keys["optimization"], optimized_cnfrs)
        _write_stage_pose_snapshot(
            lig_pdbqt=lig_pdbqt,
            cnfrs=optimized_cnfrs,
            out_sdf=os.path.join(cache_stage_dir, "optimization_poses.sdf"),
            box_center=box_center,
        )

    rescoring_cfg = cfg["rescoring"]
    use_rescore = bool(rescoring_cfg.get("enabled", False))
    rescoring_scorer = None
    rescoring_error = ""
    if use_rescore:
        try:
            rescoring_scorer = RESCORERS[rescoring_cfg["scorer"]](receptor=receptor, ligand=ligand)
        except Exception as exc:
            rescoring_scorer = None
            rescoring_error = f"rescoring_disabled: {exc}"

    pose_records: list[dict[str, Any]] = []
    cached_scores = None
    if reuse_enabled:
        cached_scores = _load_cached_scores(cache_stage_dir, stage_keys["scoring"])
        if cached_scores and len(cached_scores) == len(optimized_cnfrs):
            for i, s in enumerate(cached_scores):
                cnfr = optimized_cnfrs[i]
                pose_records.append(
                    {
                        "cnfr": cnfr,
                        "primary": float(s["primary"]),
                        "rescore": float(s["rescore"]),
                        "core_rmsd": float(s["core_rmsd"]),
                        "pose_mol": _pose_mol_from_cnfr(ligand, lig_rdkit, rd_to_od, cnfr),
                    }
                )
            reuse_info["reused_scoring"] = True
            reuse_info["notes"].append("reused:scoring")
    if not pose_records:
        for cnfr in optimized_cnfrs:
            primary_val = _score_conformation(ligand, receptor, primary_scorer, cnfr)
            rec_val = primary_val
            if rescoring_scorer is not None:
                try:
                    rec_val = _score_conformation(ligand, receptor, rescoring_scorer, cnfr)
                except Exception as exc:
                    rec_val = primary_val
                    if not rescoring_error:
                        rescoring_error = f"rescoring_failed: {exc}"

            pose_records.append(
                {
                    "cnfr": cnfr,
                    "primary": primary_val,
                    "rescore": rec_val,
                    "core_rmsd": _core_rmsd_for_pose(ligand, cnfr, lig_core_od_indices, template_core_xyz),
                    "pose_mol": _pose_mol_from_cnfr(ligand, lig_rdkit, rd_to_od, cnfr),
                }
            )
        _save_cached_scores(cache_stage_dir, stage_keys["scoring"], pose_records)

    ranked = _rank_pose_records(pose_records, use_rescore)
    selected = ranked

    best_pose_distances: list[float] = []
    if selected:
        ligand.cnfr2xyz([selected[0]["cnfr"]])
        lig_xyz = ligand.pose_heavy_atoms_coords[0, lig_core_od_indices, :].detach().numpy()
        rec_xyz = receptor.rec_heavy_atoms_xyz[anchor_rec_indices, :].detach().numpy()
        best_pose_distances = [float(np.linalg.norm(lig_xyz[i] - rec_xyz[i])) for i in range(len(anchor_rec_indices))]

    manifest = _build_mapping_manifest(
        receptor=receptor,
        template_core_match=template_core_match,
        ligand_core_match=ligand_core_match,
        ligand_core_od_indices=lig_core_od_indices,
        anchor_rec_indices=anchor_rec_indices,
        anchor_ref_distances=anchor_ref_distances,
        best_pose_distances=best_pose_distances,
        tolerance=float(core_cfg["tolerance"]),
        force_constant=float(core_cfg["force_constant"]),
        weight=float(core_cfg.get("weight", 0.0)),
    )
    if not reuse_info["notes"] and reuse_enabled:
        reuse_info["notes"].append("recompute:all_stages")
    return selected, rescoring_error, manifest, reuse_info


def _run_single_ligand(
    cfg: dict[str, Any],
    receptor_pdbqt: str,
    template_core_match: tuple[int, ...],
    template_core_xyz: np.ndarray,
    anchor_rec_indices: list[int],
    anchor_ref_distances: list[float],
    ligand_input: LigandInput,
    work_dir: str,
    cache_dir: str,
    poses_dir: str,
    manifests_dir: str,
    file_hashes: dict[str, str],
    reuse_enabled: bool,
) -> dict[str, Any]:
    ligand_id = _sanitize_id(ligand_input.ligand_id)
    ligand_work = os.path.join(work_dir, ligand_id)
    _ensure_dir(ligand_work)

    _, lig_pdbqt, lig_rdkit = _prepare_ligand_files(ligand_input, ligand_work)

    core_cfg = cfg["constraints"]["core"]
    query_smarts = cfg["inputs"].get("ligand_core_smarts") or cfg["inputs"]["reference_core_smarts"]
    patt = Chem.MolFromSmarts(query_smarts)
    if patt is None:
        raise ValueError("Invalid ligand core SMARTS")

    query_matches = list(lig_rdkit.GetSubstructMatches(patt, uniquify=False))
    if not query_matches:
        raise ValueError("No SMARTS match found in query ligand")

    max_query_mappings = int(core_cfg.get("max_query_mappings", 16))
    selected_matches = query_matches[:max_query_mappings]
    truncated = len(query_matches) > len(selected_matches)

    last_mapping_error = ""
    global_pose_records: list[dict[str, Any]] = []

    init_conf_cfg = core_cfg.get("init_conformers", {}) or {}
    init_conf_cfg_per_map = dict(init_conf_cfg)
    # Apply max_output_conformers globally across mappings, not per mapping.
    init_conf_cfg_per_map["max_output_conformers"] = None
    dedup_rmsd_threshold = float(init_conf_cfg.get("rmsd_threshold", 0.5))
    global_max_output = init_conf_cfg.get("max_output_conformers")
    global_max_output = int(global_max_output) if global_max_output is not None else None
    generated_total = 0
    mapping_local_kept_total = 0
    dedup_removed_total = 0
    cap_removed_total = 0
    candidate_pool: list[dict[str, Any]] = []
    candidate_sources: list[dict[str, int]] = []

    for map_idx, match in enumerate(selected_matches):
        if len(match) != len(template_core_xyz):
            continue

        try:
            map_lig_rdkit = lig_rdkit
            init_method = "free"
            init_warning = ""

            if bool(core_cfg.get("template_constrained_embed", True)):
                map_lig_rdkit, ok_embed, warn_embed = _constrained_embed_rdkit_to_template_core(
                    mol=map_lig_rdkit,
                    ligand_core_match=match,
                    template_core_xyz=template_core_xyz,
                    max_attempts=int(core_cfg.get("template_constrained_embed_max_attempts", 20)),
                )
                if ok_embed:
                    init_method = "constrained_embed"
                elif warn_embed:
                    init_warning = warn_embed

            if bool(core_cfg.get("rigid_prealign", True)):
                map_lig_rdkit = _rigid_align_rdkit_to_template_core(map_lig_rdkit, match, template_core_xyz)
                if init_method == "constrained_embed":
                    init_method = "constrained_embed+rigid"
                else:
                    init_method = "rigid"

            variant_mols, conf_info, conf_warn = _generate_mapping_init_conformers(
                map_lig_rdkit=map_lig_rdkit,
                ligand_core_match=match,
                template_core_xyz=template_core_xyz,
                conf_cfg=init_conf_cfg_per_map,
            )
            generated_total += int(conf_info.get("generated_count", len(variant_mols)))
            mapping_local_kept_total += int(conf_info.get("kept_count", len(variant_mols)))

            map_candidates: list[dict[str, Any]] = []
            for conf_idx, conf_mol in enumerate(variant_mols):
                run_mol = conf_mol
                if bool(core_cfg.get("rigid_prealign", True)):
                    run_mol = _rigid_align_rdkit_to_template_core(run_mol, match, template_core_xyz)
                map_candidates.append(
                    {
                        "mapping_index": map_idx,
                        "conformer_index": conf_idx,
                        "ligand_core_match": match,
                        "mol": run_mol,
                        "init_warning": " | ".join([x for x in [init_warning, conf_warn] if x]),
                    }
                )

            merged, removed = _merge_unique_candidates_by_rmsd(
                accepted=candidate_pool,
                new_candidates=map_candidates,
                rmsd_threshold=dedup_rmsd_threshold,
            )
            candidate_pool = merged
            dedup_removed_total += int(removed)
        except Exception as exc:
            last_mapping_error = str(exc)
            continue

    if global_max_output is not None and len(candidate_pool) > int(global_max_output):
        cap_removed_total = int(len(candidate_pool) - int(global_max_output))
        candidate_pool = candidate_pool[: int(global_max_output)]

    for cand in candidate_pool:
        map_idx = int(cand["mapping_index"])
        conf_idx = int(cand["conformer_index"])
        match = tuple(int(x) for x in cand["ligand_core_match"])
        run_mol = cand["mol"]

        prepared_sdf = os.path.join(ligand_work, f"{ligand_id}.map_{map_idx:03d}.conf_{conf_idx:03d}.prepared.sdf")
        map_lig_pdbqt = os.path.join(ligand_work, f"{ligand_id}.map_{map_idx:03d}.conf_{conf_idx:03d}.prepared.pdbqt")
        _write_rdkit_sdf(run_mol, prepared_sdf)
        _to_pdbqt(prepared_sdf, map_lig_pdbqt)
        candidate_sources.append({"mapping_index": int(map_idx), "conformer_index": int(conf_idx)})
        if conf_idx == 0:
            primary_sdf = os.path.join(ligand_work, f"{ligand_id}.map_{map_idx:03d}.prepared.sdf")
            primary_pdbqt = os.path.join(ligand_work, f"{ligand_id}.map_{map_idx:03d}.prepared.pdbqt")
            shutil.copyfile(prepared_sdf, primary_sdf)
            shutil.copyfile(map_lig_pdbqt, primary_pdbqt)

        stage_keys = _build_stage_keys(
            cfg=cfg,
            ligand_input=ligand_input,
            ligand_core_match=match,
            mapping_index=map_idx,
            conformer_index=conf_idx,
            file_hashes=file_hashes,
        )
        mapping_cache_dir = os.path.join(cache_dir, ligand_id, f"map_{map_idx:03d}", f"conf_{conf_idx:03d}")
        selected_records, mapping_error, mapping_manifest, reuse_info = _dock_with_mapping(
            cfg=cfg,
            receptor_pdbqt=receptor_pdbqt,
            lig_pdbqt=map_lig_pdbqt,
            lig_rdkit=run_mol,
            template_core_match=template_core_match,
            ligand_core_match=match,
            mapping_index=map_idx,
            template_core_xyz=template_core_xyz,
            anchor_rec_indices=anchor_rec_indices,
            anchor_ref_distances=anchor_ref_distances,
            cache_stage_dir=mapping_cache_dir,
            stage_keys=stage_keys,
            reuse_enabled=reuse_enabled,
        )

        if not selected_records:
            continue

        for rec in selected_records:
            row = dict(rec)
            row["mapping_index"] = int(map_idx)
            row["conformer_index"] = int(conf_idx)
            row["mapping_manifest_obj"] = mapping_manifest
            row["reuse_info"] = reuse_info
            row["mapping_error"] = mapping_error
            row["init_warning"] = str(cand.get("init_warning", "") or "")
            global_pose_records.append(row)

    if not global_pose_records:
        if last_mapping_error:
            raise RuntimeError(f"All SMARTS mapping docking attempts failed: {last_mapping_error}")
        raise RuntimeError("All SMARTS mapping docking attempts failed")

    use_rescore = bool(cfg["rescoring"].get("enabled", False))
    global_ranked = _rank_pose_records(global_pose_records, use_rescore=use_rescore)
    final_filter_cfg = cfg["outputs"].get("final_rmsd_filter", {}) or {}
    final_filter_enabled = bool(final_filter_cfg.get("enabled", False))
    final_filter_threshold = float(final_filter_cfg.get("threshold", 0.5))
    final_filter_removed = 0
    final_pool = global_ranked
    if final_filter_enabled:
        final_pool, final_filter_removed = _filter_ranked_records_by_global_rmsd(
            global_ranked,
            final_filter_threshold,
        )

    top_n = int(cfg["outputs"]["top_n"])
    final_selected = final_pool[:top_n]
    if not final_selected:
        raise RuntimeError("No poses left after final global filtering")

    out_sdf = os.path.join(poses_dir, f"{ligand_id}.sdf")
    _write_pose_records_sdf(
        pose_records=final_selected,
        out_sdf=out_sdf,
        primary_name=cfg["docking"]["primary_scorer"],
    )

    best = final_selected[0]
    best_mapping_index = int(best["mapping_index"])
    best_conformer_index = int(best["conformer_index"])
    best_manifest = best.get("mapping_manifest_obj")
    best_reuse_info = best.get("reuse_info")
    best_error_note = str(best.get("mapping_error", "") or "")
    best_init_warning = str(best.get("init_warning", "") or "")

    manifest_path = os.path.join(manifests_dir, f"{ligand_id}.mapping.yaml")
    if best_manifest is None:
        best_manifest = {"constraint_model": "unavailable", "mapping": []}
    best_manifest["ligand_id"] = ligand_id
    best_manifest["smiles"] = ligand_input.smiles
    best_manifest["mapping_index"] = int(best_mapping_index) if best_mapping_index is not None else None
    best_manifest["conformer_index"] = int(best_conformer_index) if best_conformer_index is not None else None
    best_manifest["mappings_tested"] = len(selected_matches)
    best_manifest["mappings_total"] = len(query_matches)
    best_manifest["mappings_truncated"] = bool(truncated)
    best_manifest["candidate_dedup"] = {
        "rmsd_metric": "symmetry_aware_heavy_atom_rmsd",
        "rmsd_threshold": float(dedup_rmsd_threshold),
        "max_output_conformers_global": global_max_output,
        "generated_total": int(generated_total),
        "mapping_local_kept_total": int(mapping_local_kept_total),
        "unique_candidates_before_cap": int(len(candidate_pool) + cap_removed_total),
        "unique_candidates": int(len(candidate_pool)),
        "duplicates_removed": int(dedup_removed_total),
        "cap_removed": int(cap_removed_total),
        "accepted_candidates": candidate_sources,
    }
    best_manifest["final_pose_selection"] = {
        "ranking_metric": "rescore" if use_rescore else "primary",
        "global_pose_count_before_filter": int(len(global_ranked)),
        "global_pose_count_after_filter": int(len(final_pool)),
        "top_n_requested": int(top_n),
        "top_n_written": int(len(final_selected)),
        "final_rmsd_filter": {
            "enabled": bool(final_filter_enabled),
            "metric": "symmetry_aware_heavy_atom_rmsd",
            "threshold": float(final_filter_threshold),
            "duplicates_removed": int(final_filter_removed),
        },
    }
    if best_reuse_info:
        best_manifest["reuse"] = {
            "reused_docking": bool(best_reuse_info.get("reused_docking", False)),
            "reused_optimization": bool(best_reuse_info.get("reused_optimization", False)),
            "reused_scoring": bool(best_reuse_info.get("reused_scoring", False)),
            "reuse_note": ",".join(best_reuse_info.get("notes", [])),
        }
    with open(manifest_path, "w", encoding="utf-8") as tf:
        yaml.safe_dump(best_manifest, tf, sort_keys=False)

    error_messages = []
    if truncated:
        error_messages.append(
            f"query_mapping_truncated:{len(query_matches)}->{len(selected_matches)}"
        )
    if best_error_note:
        error_messages.append(best_error_note)
    if best_init_warning:
        error_messages.append(best_init_warning)
    reuse_note = ""
    if best_reuse_info and best_reuse_info.get("notes"):
        reuse_note = ",".join(best_reuse_info["notes"])

    return {
        "ligand_id": ligand_id,
        "smiles": ligand_input.smiles,
        "status": "ok",
        "primary_score": best["primary"],
        "rescored_score": best["rescore"],
        "core_rmsd": best["core_rmsd"],
        "pose_sdf": out_sdf,
        "mapping_index": best_mapping_index,
        "conformer_index": best_conformer_index if best_conformer_index is not None else "",
        "mappings_tested": len(selected_matches),
        "mappings_total": len(query_matches),
        "mappings_truncated": str(truncated).lower(),
        "mapping_manifest": manifest_path,
        "reused_docking": str(bool(best_reuse_info and best_reuse_info.get("reused_docking", False))).lower(),
        "reused_optimization": str(bool(best_reuse_info and best_reuse_info.get("reused_optimization", False))).lower(),
        "reused_scoring": str(bool(best_reuse_info and best_reuse_info.get("reused_scoring", False))).lower(),
        "reuse_note": reuse_note,
        "error": " | ".join(error_messages),
    }


def run_from_config(config_path: str) -> None:
    cfg = _load_config(config_path)

    requested_scorers = {cfg["docking"]["primary_scorer"]}
    if cfg.get("optimization", {}).get("scorer"):
        requested_scorers.add(cfg["optimization"]["scorer"])
    if cfg["rescoring"].get("enabled", False):
        requested_scorers.add(cfg["rescoring"]["scorer"])
    if {"deeprmsd", "rmsd-vina"} & requested_scorers:
        _enable_deeprmsd_torch_load_compat()

    if cfg.get("constraints", {}).get("core", {}).get("enabled", False):
        if "max_core_rmsd" in cfg["constraints"]["core"]:
            print("[opendocker] WARNING: constraints.core.max_core_rmsd is deprecated and ignored in distance-restraint mode")

    output_dir = os.path.abspath(cfg["outputs"]["dir"])
    work_dir = os.path.join(output_dir, "work")
    cache_dir = os.path.join(work_dir, "cache")
    poses_dir = os.path.join(output_dir, "poses")
    manifests_dir = os.path.join(output_dir, "manifests")
    _ensure_dir(output_dir)
    _ensure_dir(work_dir)
    _ensure_dir(cache_dir)
    _ensure_dir(poses_dir)
    _ensure_dir(manifests_dir)
    reuse_enabled = bool(cfg["runtime"].get("reuse_enabled", True))

    receptor_input = cfg["inputs"].get("receptor_pdbqt") or cfg["inputs"].get("receptor_pdb")
    file_hashes = {
        "receptor_input": _sha256_file(receptor_input),
        "reference_core_sdf": _sha256_file(cfg["inputs"]["reference_core_sdf"]),
        "ligands_smiles": _sha256_file(cfg["inputs"]["ligands_smiles"]),
    }
    receptor_pdbqt = _prepare_receptor(receptor_input, work_dir)
    _, template_core_match, template_core_xyz = _load_reference_template(
        cfg["inputs"]["reference_core_sdf"],
        cfg["inputs"]["reference_core_smarts"],
    )
    anchor_rec_indices, anchor_ref_distances = _build_template_receptor_anchors(receptor_pdbqt, template_core_xyz)

    ligands = _read_smiles(cfg["inputs"]["ligands_smiles"])

    rows: list[dict[str, Any]] = []
    for lig in ligands:
        try:
            row = _run_single_ligand(
                cfg=cfg,
                receptor_pdbqt=receptor_pdbqt,
                template_core_match=template_core_match,
                template_core_xyz=template_core_xyz,
                anchor_rec_indices=anchor_rec_indices,
                anchor_ref_distances=anchor_ref_distances,
                ligand_input=lig,
                work_dir=work_dir,
                cache_dir=cache_dir,
                poses_dir=poses_dir,
                manifests_dir=manifests_dir,
                file_hashes=file_hashes,
                reuse_enabled=reuse_enabled,
            )
        except Exception as exc:
            row = {
                "ligand_id": _sanitize_id(lig.ligand_id),
                "smiles": lig.smiles,
                "status": "failed",
                "primary_score": np.nan,
                "rescored_score": np.nan,
                "core_rmsd": np.nan,
                "pose_sdf": "",
                "mapping_index": "",
                "conformer_index": "",
                "mappings_tested": 0,
                "mappings_total": 0,
                "mappings_truncated": "false",
                "mapping_manifest": "",
                "reused_docking": "false",
                "reused_optimization": "false",
                "reused_scoring": "false",
                "reuse_note": "",
                "error": str(exc),
            }
        rows.append(row)

    summary_path = os.path.join(output_dir, "summary.csv")
    fields = [
        "ligand_id",
        "smiles",
        "status",
        "primary_score",
        "rescored_score",
        "core_rmsd",
        "pose_sdf",
        "mapping_index",
        "conformer_index",
        "mappings_tested",
        "mappings_total",
        "mappings_truncated",
        "mapping_manifest",
        "reused_docking",
        "reused_optimization",
        "reused_scoring",
        "reuse_note",
        "error",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[opendocker] Finished. Summary: {summary_path}")
