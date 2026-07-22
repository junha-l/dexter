# DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-1b3d6d.svg)](https://junha-l.github.io/dexter/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.16046-b31b1b.svg)](https://arxiv.org/abs/2601.16046)
[![Project Page](https://img.shields.io/badge/Project-Page-1f9bcf.svg)](https://junha-l.github.io/dexter/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-ffce44.svg)](https://huggingface.co/datasets/EunhaPark/project_dexter)

**[Junha Lee](https://junha-l.github.io/)<sup>1</sup>, &nbsp; [Eunha Park](https://dmsgk724.github.io/)<sup>1</sup>, &nbsp; [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/)<sup>1,2</sup>**

<sup>1</sup>POSTECH &nbsp;&nbsp; <sup>2</sup>RLWRLD

> DextER introduces contact-based **embodied reasoning** for language-driven dexterous grasp generation. Given a 3D object and a natural-language instruction, DextER autoregressively predicts *which finger links contact where* on the object surface before generating the final 28-DoF Shadow Hand grasp.
>
> **Core idea:** instead of regressing a grasp directly, DextER reasons about contact intent first — an explicit *contact → action* chain that grounds the instruction in object geometry, yielding more physically plausible, language-faithful grasps and stronger generalization to unseen objects and grasp types.

## Updates
- **2026-07-22** — Checkpoint release; hotfix in the dataset.
- **2026-06-03** — Initial code release: training, evaluation, and benchmarking pipelines are now public.

## Setup

> Tested with PyTorch 2.8.0 on CUDA 12.8.

Install with [`uv`](https://docs.astral.sh/uv/)

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install everything (incl. the CUDA/source-built deps)
uv sync
source .venv/bin/activate
```

Experiment tracking uses Weights & Biases. Run `wandb login` then `wandb init`, or set `wandb.enabled=false` to disable it.

## Data Preparation

> [!WARNING]
> **Re-download if you pulled the dataset before 2026-07-22.** While packaging the preprocessed data for the Hugging Face Hub we found a data error: for some scenes with **≥10 grasps**, the grasp poses were misaligned with their contact-point annotations. This is now **fixed and republished** — delete any earlier copy and re-download from the Hub.

DextER is trained and evaluated on two dexterous-grasping datasets:

- **DexGYS** ([*Grasp as You Say*](https://arxiv.org/abs/2405.19291), NeurIPS 2024) — the primary benchmark for instruction-conditioned grasp generation.
- **Dexonomy** ([*Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy*](https://arxiv.org/abs/2504.18829), RSS 2025) — a grasp-taxonomy dataset for measuring generalization to unseen objects and grasp types.

We release both datasets fully preprocessed on the Hugging Face Hub, so you can skip preprocessing entirely:

```bash
# Downloads the preprocessed data (~121 GB) as dexgys.tar.gz + dexonomy.tar.gz into datasets/
hf download --repo-type dataset EunhaPark/project_dexter --local-dir datasets

# Extract each tarball to a dataset root of your choice (-C is the destination dir).
# Set DATA_ROOT to wherever you want the data to live; the archives unpack to
# `dexgys_final/` and `dexonomy_final/` under it.
DATA_ROOT=/path/to/datasets
mkdir -p "$DATA_ROOT"
tar -xvzf datasets/dexgys.tar.gz   -C "$DATA_ROOT"
tar -xvzf datasets/dexonomy.tar.gz -C "$DATA_ROOT"
```

This gives you `$DATA_ROOT/dexgys_final` and `$DATA_ROOT/dexonomy_final`. Point training and evaluation at these roots via `data.path=...` and `--data-dir` respectively; extract into `/root/data` to match the config defaults and skip the override.

## Model Zoo

We release pretrained DextER checkpoints on the Hugging Face Hub. Each is a Qwen2.5-0.5B backbone trained on one of the two benchmarks. Download a checkpoint directory and pass it to `scripts/test.py` via `--checkpoint-dir` (see [Evaluation](#evaluation)).

| Model | Backbone | Training data | Checkpoint |
|---|---|---|---|
| DextER-DexGYS | Qwen2.5-0.5B | DexGYS | [🤗 junhalee/dexter-qwen2.5-0.5B-dexgys](https://huggingface.co/junhalee/dexter-qwen2.5-0.5B-dexgys) |
| DextER-Dexonomy | Qwen2.5-0.5B | Dexonomy | [🤗 junhalee/dexter-qwen2.5-0.5B-dexonomy](https://huggingface.co/junhalee/dexter-qwen2.5-0.5B-dexonomy) |

```bash
# Download a checkpoint into checkpoints/ and evaluate it
hf download junhalee/dexter-qwen2.5-0.5B-dexgys --local-dir checkpoints/dexter-qwen2.5-0.5B-dexgys

python scripts/test.py \
    --checkpoint-dir checkpoints/dexter-qwen2.5-0.5B-dexgys \
    --data-dir $DATA_ROOT/dexgys_final \
    --save-pred
```

## Training

Training is configured with **Hydra**; compose a run by selecting a `model=` and `data=` config.

```bash
# Default run
python scripts/train.py experiment_name=dexter

# Train model variant
python scripts/train.py \
    model=dexter_1.5B \
    experiment_name=dexter_1.5B

# Point at the dataset root you extracted to ($DATA_ROOT from Data Preparation).
# `data.path=` overrides the config's default `path:`; omit it only if you
# extracted into /root/data (the config default).
python scripts/train.py \
    data.path=$DATA_ROOT/dexgys_final \
    experiment_name=dexter

# Train on Dexonomy
python scripts/train.py \
    data=dexonomy \
    data.path=$DATA_ROOT/dexonomy_final \
    experiment_name=dexter_dexonomy

# Override any hyperparameter
python scripts/train.py \
    training.batch_size=16 \
    training.learning_rate=2e-4 \
    experiment_name=dexter_bs16x1_lr2e-4

# Multi-GPU (8) with accelerate
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    scripts/train.py \
    experiment_name=dexter_bs8x8
```

Checkpoints are written to `checkpoints/<experiment_name>/` with a `config.yaml` for reproducibility.

## Evaluation

Evaluate a checkpoint with `scripts/test.py`. Constrained decoding (generation restricted to valid action-token bins) and ECoT parsing are **on by default**; disable them with `--noconstrain-to-actions` / `--noparse-ecot`. `--save-pred` writes a `predictions.json` for benchmarking:

```bash
# DexGYS, constrained decoding (default)
python scripts/test.py \
    --checkpoint-dir <ckpt_dir> \
    --data-dir $DATA_ROOT/dexgys_final \
    --save-pred

# Dexonomy generalization (--split: seen_val|unseen_grasp|unseen_obj|unseen_both)
python scripts/test.py \
    --checkpoint-dir <ckpt_dir> \
    --data-dir $DATA_ROOT/dexonomy_final \
    --split unseen_obj \
    --save-pred

# Partial RGB-D robustness (+ sensor noise)
python scripts/test.py \
    --checkpoint-dir <ckpt_dir> \
    --data-dir $DATA_ROOT/dexgys_final \
    --save-pred \
    --partial-obs \
    --partial-obs-add-noise

# Contact-guided steering
python scripts/test.py \
    --checkpoint-dir <ckpt_dir> \
    --data-dir $DATA_ROOT/dexonomy_final \
    --split seen_val \
    --save-pred \
    --steer-link-num 3
```

Each run writes a `predictions.json` (plus `test.log`) under `test_output/<checkpoint-parent>_<checkpoint-name><postfix>/`. 

Dexonomy measures generalization, so generate one `predictions.json` per split — `seen_val`, `unseen_grasp`, `unseen_obj`, `unseen_both`.

The benchmark guides below refer to this directory as `$PRED`:

```bash
PRED=test_output/dexgys_dexter_test
```

## Benchmarking

Both benchmarks score a `predictions.json` from the step above with two kinds of metric: **computational** metrics evaluated directly from the predicted poses, and a **simulation-based** success rate measured by rolling the grasp out in a physics engine. Follow each benchmark's README for detailed environment setup and step-by-step scoring instructions:

- [DexGYS](benchmark/dexgys/README.md)
- [Dexonomy](benchmark/dexonomy/README.md)

## Citation

If you find DextER useful, please cite:

```bibtex
@inproceedings{lee2026dexter,
    title     = {DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning},
    author    = {Lee, Junha and Park, Eunha and Cho, Minsu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```

## Acknowledgements

DextER builds on the following works. We thank the authors for releasing their code and data:

- [DexGYS](https://github.com/iSEE-Laboratory/Grasp-as-You-Say)
- [Dexonomy](https://github.com/JYChen18/Dexonomy)
- [DexGraspBench](https://github.com/JYChen18/DexGraspBench)
- [Uni3D](https://github.com/baaivision/Uni3D)
- [PartField](https://github.com/nv-tlabs/PartField)
