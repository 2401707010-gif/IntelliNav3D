# IntelliNav3D: Setup, Configuration, Data, and Evaluation

This guide describes how to install the IntelliNav3D code, prepare the required data, configure experiments, and run the evaluation pipeline. Check the repository's `README.md` and each script's `--help` output for project-specific details.

## 1. Environment and dependencies

A clean Python 3.10 environment is a reasonable starting point. The compatible PyTorch build depends on your operating system and GPU/CUDA version.

```bash
conda create -n intellinav3d python=3.10 -y
conda activate intellinav3d

# Install the PyTorch build appropriate for your computer.
# Choose the correct command from https://pytorch.org/get-started/locally/
pip install torch

pip install -r requirements.txt
pip install -e .
```

Run the available tests and smoke test before starting a full experiment:

```bash
python -m pytest -q tests/
python scripts/selftest.py
```

If a command or script is not present in the repository, check the actual repository contents and README before using it. A smoke test is not a substitute for reproducing the full experiments.

## 2. Dataset access and preparation

The data pipeline may require the 3D-FRONT scene data and 3D-FUTURE model assets. Obtain these from the official dataset portal:

- Official portal: https://tianchi.aliyun.com/dataset/65347

After downloading, follow the directory layout expected by the project's data loader. For the default layout documented for this project, that may look like:

```text
data/
  raw/
    3D-FRONT/
      *.json
    3D-FUTURE-model/
      model_info.json
      <model-id>/
        raw_model.obj
```

Then inspect the available preparation scripts and their options:

```bash
python scripts/prepare_data.py --help
python scripts/make_split.py --help
```

Run the preparation commands supported by the repository after setting the correct input and output paths.

**Datasets and third-party assets are not included in this code package.** Download access may require registration or acceptance of the provider's terms. Follow all dataset licensing, attribution, and redistribution restrictions. Do not commit raw datasets or model assets to GitHub or Zenodo unless their terms explicitly allow redistribution.

## 3. Configuration

The default experiment configuration is expected at `configs/default.yaml`. Review it before running experiments; confirm that data paths, model settings, random seeds, output directories, and evaluation parameters match your environment.

To preserve the tracked defaults, you can make a local copy:

```bash
cp configs/default.yaml configs/local.yaml
```

Use the local configuration only if the relevant scripts support a configuration argument. Check the actual options first:

```bash
python scripts/run_experiments.py --help
```

For reproducibility, record all configuration changes, software versions, hardware details, random seeds, and dataset versions. Do not describe implementation choices as paper-specified settings unless the manuscript actually specifies them.

## 4. Evaluation

First inspect the scripts and their accepted arguments:

```bash
python scripts/run_experiments.py --help
python scripts/evaluate_external.py --help
python scripts/make_tables.py --help
python scripts/make_figures.py --help
```

Run the experiment stages and post-processing commands supported by the codebase. If the project provides stage names such as `main`, `ablation`, `plausibility`, `heldout`, or `sensitivity`, use those exact values as shown by the script's help output.

Full experiments may require significant compute time, GPU memory, and storage. Keep generated outputs separate from source code, and report only metrics obtained from runs that were actually completed. Do not present planned or unexecuted experiments as measured results.

## 5. Repository contents

The repository should include, where applicable:

- `intellinav3d/`: framework source code.
- `configs/`: default and example experiment configuration files.
- `scripts/`: data preparation, training/inference, evaluation, and result-generation scripts.
- `tests/`: automated tests and smoke tests.
- `requirements.txt` and/or `pyproject.toml`: dependency and package metadata.
- `README.md`: project overview and quick-start instructions.
- `LICENSE`: license for the repository code.

Update this list if the actual repository has different filenames or directories.

## 6. License and third-party assets

The `LICENSE` file in this repository applies to the code only, unless a file explicitly states otherwise. It does not automatically grant rights to third-party datasets, 3D models, external baseline implementations, pretrained weights, or other separately licensed materials. Review and comply with the relevant terms before using or redistributing those assets.
