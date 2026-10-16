[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# A Supervised Learning-based Branching Method in Branch-and-Price

This archive is distributed in association with the
[INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE). 

The software and data in this repository are a snapshot of the software and data that were used in the research reported on in the paper [A Supervised Learning-based Branching Method in Branch-and-Price](https://doi.org/10.1287/ijoc.2025.1318)
by Baiyang He, Ying Meng, Lixin Tang, and Jiyin Liu.

## Cite

To cite the contents of this repository, please cite both the paper and this
code repository using their respective DOIs:

- Paper: https://doi.org/10.1287/ijoc.2025.1318
- Code: https://doi.org/10.1287/ijoc.2025.1318.cd

BibTeX for the code repository:

```bibtex
@misc{He2026,
  author =    {Baiyang He and Ying Meng and Lixin Tang and Jiyin Liu},
  publisher = {INFORMS Journal on Computing},
  title =     {A Supervised Learning-based Branching Method in Branch-and-Price},
  year =      {2026},
  doi =       {10.1287/ijoc.2025.1318.cd},
  url =       {https://github.com/INFORMSJoC/2025.1318},
  note =      {Available for download at https://github.com/INFORMSJoC/2025.1318},
}
```

## Description

This repository provides a reproducible C++17/Python implementation of the
General Batching Decision (GBD) model and the SLbJNV method presented in the
paper. It includes the GBD set-packing master model, machine/master-item pricing
problems, supervised-learning workflow, and integrated branch-and-price solver.

## Software overview

The project includes the configurable GBD generator and validators plus a CPLEX-backed RMP, binary
pricing problems, convergent column generation, and a baseline DFS
branch-and-price search. All five structural branching rules generate candidates
from the current LP solution and propagate their child constraints through the
RMP and pricing layer. The data-collection mode uses DFS and strong
branching to record complete trees, computes Equation (9) labels after the
search, builds key-node sequences and 1000-dimensional feature vectors, and
writes ordered JSON Lines datasets with an instance-level 70:15:15 split.
The PyTorch implementation contains the 1000-to-16 embedding, attention-based
Seq2Seq key-sequence model, three-layer score MLP, Log-cosh sequence loss,
ordered cumulative score error, random D2 sampling, projected adaptive beta
updates, validation, early stopping, checkpoints, and persistent batch prediction.
The experiment driver supports baseline, complete-tree data collection, and
joint SLbJNV modes over one file or a directory, with TXT/CSV summaries and
separate incumbent and bound histories.

## Architecture

- `cpp/include/slbjnv/instance`: GBD items, machines, compatibility and instance validation.
- `cpp/include/slbjnv/model`: feasible batch columns and solver-independent LP/MIP contracts.
- `cpp/include/slbjnv/column_generation`: RMP, pricing and column-generation interfaces.
- `cpp/include/slbjnv/branching`: the five structural branching-variable types and inherited constraints.
- `cpp/include/slbjnv/branch_and_price`: nodes, policies, incumbent state, tree and solver orchestration.
- `cpp/include/slbjnv/features`: the single 1000-dimensional feature contract used offline and online.
- `cpp/include/slbjnv/data_collection`: ordered JSONL sample and complete-tree record contracts.
- `cpp/include/slbjnv/python_interface`: persistent batch-prediction interface.
- `python/slbjnv`: dataset, network, loss, training, evaluation and prediction APIs.
- `configs`: all tunable instance, B&P and learning defaults.

## Build in CLion

Requirements: CMake 3.21+, a C++17 compiler, Python 3.10+, and PyTorch. CPLEX
is the preferred backend; HiGHS is the supported open-source fallback.

```text
cmake -S . -B build -DSLBJNV_ENABLE_CPLEX=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

On Windows the CPLEX backend loads the Callable Library DLL dynamically, so it
works with MinGW without linking the MSVC-only Concert C++ libraries. 

For HiGHS use `-DSLBJNV_ENABLE_HIGHS=ON`. Embedded inference additionally
uses `-DSLBJNV_ENABLE_PYTHON=ON` and a discoverable pybind11 installation.

## Executable modes

Run the baseline solver with:

```text
build/run_experiment --instance tests/fixtures/gbd_2_5_fractional.txt
```

Generate supervised-learning data with a CPLEX-enabled build:

```text
build/generate_training_data --instances data/instances --output data/training --config configs/branch_and_price.yaml
```

The command writes `train.txt`, `validation.txt`, `dataset_summary.txt`, and a
JSON Lines tree file for each solved instance under `data/training/trees`.
Samples belonging to the same instance remain in one split and preserve their
`decision_index` order. Labels are emitted only for instances solved to proven
optimality.

## Train and evaluate SLbJNV

Activate the Python environment in which the project was installed, then run:

```text
python -m slbjnv.train --config configs/slbjnv.yaml
python -m slbjnv.evaluate --model models/best_model.pt --data data/training/validation.txt --device cuda
```

With `training.device: auto`, CUDA is selected whenever PyTorch reports an
available GPU. 

## Build and run embedded SLbJNV inference

Install the optional embedding dependency in the same environment as PyTorch:

```text
python -m pip install -e ".[embed]"
```

Configure CMake with CPLEX, Python, and the pybind11 CMake package from that
environment. On Windows/PowerShell:

```text
$pythonExe = (Get-Command python).Source
$pybindDir = python -m pybind11 --cmakedir
cmake -S . -B build-python -DSLBJNV_ENABLE_CPLEX=ON -DSLBJNV_ENABLE_PYTHON=ON `
  -DPython_EXECUTABLE="$pythonExe" -Dpybind11_DIR="$pybindDir"
cmake --build build-python
```

Run genuine joint node-variable selection with one persistent Python model and
batched CUDA inference:

```text
build-python/run_experiment --instance tests/fixtures/gbd_2_5_fractional.txt `
  --mode slbjnv --model models/best_model.pt --device cuda
```

## Run experiments

Run baseline and SLbJNV on every TXT instance in a directory:

```text
build-python/run_experiment --instances data/instances --mode all `
  --model models/best_model.pt --device cuda `
  --config configs/branch_and_price.yaml --output-dir results/gbd_comparison
```

Use `--mode baseline`, `--mode collect`, or `--mode slbjnv` for one method.
The collection mode also writes ordered samples and complete trees below the
selected result directory. Optional reference objectives use a whitespace
separated file with one `instance_id objective` pair per line:

```text
build-python/run_experiment --instances data/instances --mode all `
  --model models/best_model.pt --device cuda `
  --reference-file data/reference_objectives.txt
```

Each run creates `experiment_results.csv`, `experiment_results.txt`, and
per-instance CSV histories under `histories/`. Reference-objective attainment
and optimality certification are reported as distinct metrics.

## Generate GBD instances

After building, run from the repository root:

```text
build/generate_instances --config configs/instance_generator.yaml
```

The default configuration creates ten instances for each of the six requested
machine/item sizes. With `seed: null` a fresh seed is selected and printed;
`--seed UINT64` makes the complete output reproducible. Every written file is
read back by the C++ parser immediately. The independent validator is:

```text
python scripts/validate_instances.py --config configs/instance_generator.yaml data/instances
```

The versioned text format is documented in `docs/instance_format.md`.

