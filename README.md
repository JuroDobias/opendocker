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
- for each mapping, apply flat-bottom quadratic restraint with:
  - `tolerance` (half-width)
  - `force_constant` (penalty strength)
- select best mapping by primary docking score

## Inputs

`config.yaml` controls receptor input, SMILES batch, scorer/sampler, optimization, rescoring, and constraint parameters.
By default, runs reuse cached intermediate stages in `outputs.dir/work/cache` when relevant settings and input file contents are unchanged.
Set `runtime.reuse_enabled: false` to force full recomputation.

For receptor input you can use either:
- `inputs.receptor_pdbqt` (preferred, no conversion)
- `inputs.receptor_pdb` (converted to PDBQT)

See `config.example.yaml` for the schema.

## Notes

Machine-specific environment and path notes should go to `AGENTS.MD`, not to this README.
