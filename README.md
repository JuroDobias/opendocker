# opendocker

YAML-driven docking wrapper around OpenDock.

## Installation (Conda)

Create environment from [`enviroment.yaml`](enviroment.yaml):

```bash
git clone https://github.com/JuroDobias/opendocker.git
git clone https://github.com/guyuehuo/opendock.git
cd opendocker
conda env create -f enviroment.yaml
conda activate opendocker
pip install -e ../opendock --no-build-isolation
pip install -e . --no-build-isolation
```

## Run

```bash
opendocker run -c config.yaml
```

## Run without install (optional)

```bash
python -m opendocker run -c config.yaml
# or
./bin/opendocker run -c config.yaml
```

## Core Constraint Mode

Current core constraint is based on template-derived ligand-protein distance restraints:
- match SMARTS on template ligand (`reference_core_sdf`)
- for each matched template atom, pick nearest receptor heavy atom and store reference distance
- match SMARTS on query ligand; try up to `max_query_mappings` query matches
- initialize each mapping from template-guided geometry (`template_constrained_embed`) with rigid core pre-alignment (`rigid_prealign`)
- optional RDKit multi-conformer initialization with greedy RMSD filtering (`constraints.core.init_conformers`) per mapping
- merge all accepted mapping/conformer candidates into a global pose pool
- for each mapping, apply flat-bottom quadratic restraint with:
  - `tolerance` (half-width)
  - `force_constant` (penalty strength)
- rank poses globally across all candidates and keep `outputs.top_n`
- optional final global symmetry-aware heavy-atom RMSD filtering (`outputs.final_rmsd_filter`)
- optional final score-window filtering before `top_n` (`outputs.final_score_filter`)

## Inputs

`config.yaml` controls receptor input, SMILES batch, scorer/sampler, optimization, rescoring, and constraint parameters.
By default, runs reuse cached intermediate stages in `outputs.dir/work/cache` when relevant settings and input file contents are unchanged.
Set `runtime.reuse_enabled: false` to force full recomputation.
Set `docking.enabled: false` to skip sampling/docking and run only optimization/rescoring from the prepared input pose.
When `docking.enabled: false`, optimization defaults to `freeze_rigid_body: true` so only internal torsions are optimized.

For receptor input you can use either:
- `inputs.receptor_pdbqt` (preferred, no conversion)
- `inputs.receptor_pdb` (converted to PDBQT)

See `config.example.yaml` for the schema.

## Scoring Support

The following scorers are supported in this repository setup:
- `vina`
- `deeprmsd`
- `rmsd-vina`

The following OpenDock scorers are currently **not supported out-of-the-box** here:
- `zranker`
- `sfct`

They depend on external pipelines with hardcoded paths in upstream OpenDock, so they typically fail or return fallback scores unless those external tools/paths are reconfigured.

## Notes

Machine-specific environment and path notes should go to `AGENTS.MD`, not to this README.
