import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

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

    original_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        try:
            return original_load(f, *args, **kwargs)
        except Exception as exc:
            path = str(f)
            msg = str(exc)
            if "deeprmsd_model" in path and "Weights only load failed" in msg:
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
    sampler = _must_get(docking, "sampler")
    if primary not in PRIMARY_SCORERS:
        raise ValueError(f"Unsupported docking.primary_scorer: {primary}")
    if sampler not in SAMPLERS:
        raise ValueError(f"Unsupported docking.sampler: {sampler}")

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
        if float(core["tolerance"]) < 0:
            raise ValueError("constraints.core.tolerance must be >= 0")
        if float(core["force_constant"]) < 0:
            raise ValueError("constraints.core.force_constant must be >= 0")
        if int(core["max_query_mappings"]) < 1:
            raise ValueError("constraints.core.max_query_mappings must be >= 1")

    _must_get(runtime, "exhaustiveness")
    _must_get(runtime, "num_modes")
    _must_get(outputs, "dir")
    _must_get(outputs, "top_n")

    return cfg


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    return _validate_config(cfg)


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def _post_optimize(
    cnfrs: list[torch.Tensor],
    ligand: LigandConformation,
    scoring_function: BaseScoringFunction,
    minimizer_name: str,
    steps: int,
    lr: float,
) -> list[torch.Tensor]:
    minimizer = MINIMIZERS[minimizer_name]
    if minimizer is None:
        return cnfrs

    optimized = []
    for cnfr in cnfrs:
        x = [torch.tensor(cnfr.detach().numpy(), dtype=torch.float32).requires_grad_()]

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
    template_core_xyz: np.ndarray,
    anchor_rec_indices: list[int],
    anchor_ref_distances: list[float],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
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

    sampler_name = cfg["docking"]["sampler"]
    sampler_cls = SAMPLERS[sampler_name]
    step_factor = SAMPLER_STEP_FACTOR[sampler_name]
    nsteps = int(cfg["runtime"]["exhaustiveness"]) * step_factor * int(ligand.number_of_heavy_atoms)

    opt_cfg = cfg["optimization"]
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

    optimized_cnfrs = clustered_cnfrs
    if opt_cfg.get("enabled", False):
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

    pose_records = []
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
            }
        )

    ranked = _rank_pose_records(pose_records, use_rescore)
    top_n = int(cfg["outputs"]["top_n"])
    selected = ranked[:top_n]

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
    return selected, rescoring_error, manifest


def _run_single_ligand(
    cfg: dict[str, Any],
    receptor_pdbqt: str,
    template_core_match: tuple[int, ...],
    template_core_xyz: np.ndarray,
    anchor_rec_indices: list[int],
    anchor_ref_distances: list[float],
    ligand_input: LigandInput,
    work_dir: str,
    poses_dir: str,
    manifests_dir: str,
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

    best_mapping_index = None
    best_primary = None
    best_selected_records = None
    best_error_note = ""
    best_manifest: dict[str, Any] | None = None
    last_mapping_error = ""

    for map_idx, match in enumerate(selected_matches):
        if len(match) != len(template_core_xyz):
            continue

        try:
            selected_records, mapping_error, mapping_manifest = _dock_with_mapping(
                cfg=cfg,
                receptor_pdbqt=receptor_pdbqt,
                lig_pdbqt=lig_pdbqt,
                lig_rdkit=lig_rdkit,
                template_core_match=template_core_match,
                ligand_core_match=match,
                template_core_xyz=template_core_xyz,
                anchor_rec_indices=anchor_rec_indices,
                anchor_ref_distances=anchor_ref_distances,
            )
        except Exception as exc:
            last_mapping_error = str(exc)
            continue

        if not selected_records:
            continue

        mapping_primary = min(float(x["primary"]) for x in selected_records)
        if best_primary is None or mapping_primary < best_primary:
            best_primary = mapping_primary
            best_mapping_index = map_idx
            best_selected_records = selected_records
            best_error_note = mapping_error
            best_manifest = mapping_manifest

    if best_selected_records is None:
        if last_mapping_error:
            raise RuntimeError(f"All SMARTS mapping docking attempts failed: {last_mapping_error}")
        raise RuntimeError("All SMARTS mapping docking attempts failed")

    best_cnfrs = [x["cnfr"] for x in best_selected_records]

    # Rebuild ligand object for trajectory export from selected conformations.
    export_ligand = LigandConformation(lig_pdbqt)
    box_center = np.mean(template_core_xyz, axis=0).tolist()
    export_ligand.ligand_center[0][0] = box_center[0]
    export_ligand.ligand_center[0][1] = box_center[1]
    export_ligand.ligand_center[0][2] = box_center[2]

    temp_pdbqt = os.path.join(ligand_work, f"{ligand_id}.poses.pdbqt")
    out_sdf = os.path.join(poses_dir, f"{ligand_id}.sdf")
    write_ligand_traj(
        best_cnfrs,
        export_ligand,
        temp_pdbqt,
        information={
            f"primary_{cfg['docking']['primary_scorer']}": [x["primary"] for x in best_selected_records],
            "rescore": [x["rescore"] for x in best_selected_records],
            "core_rmsd": [x["core_rmsd"] for x in best_selected_records],
        },
    )
    _to_pdbqt(temp_pdbqt, out_sdf)

    manifest_path = os.path.join(manifests_dir, f"{ligand_id}.mapping.yaml")
    if best_manifest is None:
        best_manifest = {"constraint_model": "unavailable", "mapping": []}
    best_manifest["ligand_id"] = ligand_id
    best_manifest["smiles"] = ligand_input.smiles
    best_manifest["mapping_index"] = int(best_mapping_index) if best_mapping_index is not None else None
    best_manifest["mappings_tested"] = len(selected_matches)
    best_manifest["mappings_total"] = len(query_matches)
    best_manifest["mappings_truncated"] = bool(truncated)
    with open(manifest_path, "w", encoding="utf-8") as tf:
        yaml.safe_dump(best_manifest, tf, sort_keys=False)

    best = best_selected_records[0]
    error_messages = []
    if truncated:
        error_messages.append(
            f"query_mapping_truncated:{len(query_matches)}->{len(selected_matches)}"
        )
    if best_error_note:
        error_messages.append(best_error_note)

    return {
        "ligand_id": ligand_id,
        "smiles": ligand_input.smiles,
        "status": "ok",
        "primary_score": best["primary"],
        "rescored_score": best["rescore"],
        "core_rmsd": best["core_rmsd"],
        "pose_sdf": out_sdf,
        "mapping_index": best_mapping_index,
        "mappings_tested": len(selected_matches),
        "mappings_total": len(query_matches),
        "mappings_truncated": str(truncated).lower(),
        "mapping_manifest": manifest_path,
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
    poses_dir = os.path.join(output_dir, "poses")
    manifests_dir = os.path.join(output_dir, "manifests")
    _ensure_dir(output_dir)
    _ensure_dir(work_dir)
    _ensure_dir(poses_dir)
    _ensure_dir(manifests_dir)

    receptor_input = cfg["inputs"].get("receptor_pdbqt") or cfg["inputs"].get("receptor_pdb")
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
                poses_dir=poses_dir,
                manifests_dir=manifests_dir,
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
                "mappings_tested": 0,
                "mappings_total": 0,
                "mappings_truncated": "false",
                "mapping_manifest": "",
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
        "mappings_tested",
        "mappings_total",
        "mappings_truncated",
        "mapping_manifest",
        "error",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[opendocker] Finished. Summary: {summary_path}")
