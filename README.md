# opendocker

YAML-driven docking wrapper around OpenDock.

## Option 1: repo-local command (no install)

```bash
export PATH="$(pwd)/bin:$PATH"
opendocker run -c config.yaml
```

## Option 2: install CLI into env

```bash
pip install -e . --no-build-isolation
```

Then run:

```bash
opendocker run -c config.yaml
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

See `config.example.yaml` for the schema.

## Notes

Machine-specific environment and path notes should go to `AGENTS.MD`, not to this README.
