# TTA_adv (ICLR 2025)

## Paper

**[On the Adversarial Vulnerability of Label-Free Test-Time Adaptation](https://openreview.net/forum?id=N0ETIi580T)**  
ICLR 2025 • Shahriar Rifat, Jonathan Ashdown, Michael J. De Lucia, Ananthram Swami, Francesco Restuccia  
Links: [OpenReview](https://openreview.net/forum?id=N0ETIi580T) • [PDF](https://openreview.net/pdf?id=N0ETIi580T)

This repo runs TTA on corrupted datasets (CIFAR-10-C / CIFAR-100-C / ImageNet-C) and can optionally craft attacks that poison a portion of each test batch to degrade adaptation.

## What’s inside

- **TTA algorithms** (see `tta_algo/`): `tent`, `eata`, `sar`, `sotta`, `note`, `cotta`
- **Attacks** (see `tta_attack/`):
  - `dia`: gradient-based batch poisoning (cross-entropy on benign samples)
  - `fca`: feature-based batch poisoning

## Citation

If you use this code in your research, please cite the paper.

```bibtex
@inproceedings{tta_adv_iclr2025,
  title        = {On the Adversarial Vulnerability of Label-Free Test-Time Adaptation},
  author       = {Shahriar Rifat and Jonathan Ashdown and Michael J. De Lucia and Ananthram Swami and Francesco Restuccia},
  booktitle    = {International Conference on Learning Representations (ICLR)},
  year         = {2025},
  url          = {https://openreview.net/forum?id=N0ETIi580T}
}
```

## Repository layout

- `main.py`: main entrypoint (loads model/data, runs evaluation, writes result files)
- `config/conf.py`: defaults (dataset, batch size, corruption severity, attack/TTA hyperparams)
- `data/data.py`: loaders for CIFAR-*-C (via RobustBench) and ImageNet-C (folder-based)
- `tta_algo/`: implementations of TTA adapters
- `tta_attack/`: attack implementations
- `corrupted_data/`: expected location for CIFAR-*-C and ImageNet-C
- `run*.sh`: example sweep scripts

## Setup

### Python

Python **3.10+** is recommended.

### Install dependencies

A minimal install that matches the imports used in the repo:

```bash
pip install -U \
  torch torchvision \
  timm robustbench \
  yacs tqdm pandas numpy \
  pillow matplotlib
```

Notes:
- `robustbench` is used to load CIFAR-10-C / CIFAR-100-C arrays.
- ImageNet models are loaded via `timm`.

## Data

### CIFAR-10-C / CIFAR-100-C

`data/data.py` loads CIFAR-C through RobustBench and expects the corruption files under `cfg.DATA_DIR` (default: `corrupted_data/`).

Your directory should look like:

- `corrupted_data/CIFAR-10-C/*.npy`
- `corrupted_data/CIFAR-100-C/*.npy`

### ImageNet-C

For ImageNet-C, the loader uses an `ImageFolder` layout:

- `corrupted_data/ImageNet-C/<corruption>/<severity>/<class_folder>/*`

Example:

- `corrupted_data/ImageNet-C/brightness/3/n01440764/*.JPEG`

## Running

All runs go through `main.py`.

### Common flags

- `--attack`: `dia` | `fca`
- `--tta`: `tent` | `eata` | `sar` | `sotta` | `rotta` | `note` | `cotta`
- `--dataset`: `cifar10c` | `cifar100c` | `imagenetc`
- `--severity`: corruption severity (1–5)
- `--batch_size`: batch size used for evaluation
- `--gpu_id`: CUDA device id (if available)

### Examples

CIFAR-10-C + Tent + DIA:

```bash
python3 main.py --dataset cifar10c --severity 3 --tta tent --attack dia --batch_size 200 --gpu_id 0
```

CIFAR-100-C + EATA + FCA:

```bash
python3 main.py --dataset cifar100c --severity 3 --tta eata --attack fca --batch_size 200 --gpu_id 0
```

## Outputs

`main.py` writes a tab-separated results file per run in the working directory:

- `results_<tta>_<attack>_<dataset>_<batch_size>_<severity>`


## Configuration

Defaults live in `config/conf.py`. Key knobs:

- `cfg.DIA.MAL_PORTION`: fraction of each batch treated as malicious
- `cfg.DIA.EPS`, `cfg.DIA.ALPHA`, `cfg.DIA.STEPS`: attack strength/iterations
- `cfg.DIA.ADV_MODEL`: whether the model expects inputs already normalized

