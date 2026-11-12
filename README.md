# Gated Semantic–Temporal Fusion for Detecting AI-Generated Natural Disaster Videos

Code, data splits and result files for the paper

> **Gated semantic–temporal fusion for detecting AI-generated natural disaster videos on social media under a shortcut-controlled protocol**
> Seda İşgüzar, Onur Polat, Özgür Tonkal, Alper Talha Karadeniz, Kazım Kılıç, Zafer Cömert, Muammer Türkoğlu
> *Submitted to Expert Systems with Applications.*

The repository contains:

- the **shortcut-controlled protocol**: border/letterbox removal, a 10% margin crop and an aspect-preserving 224×224 crop, near-duplicate-aware group five-fold cross-validation and leave-one-disaster-out (LODO) splits;
- the **proposed model**: a frozen multi-layer CLIP ViT-L/14 semantic stream and a ResNet-18 frame-difference temporal stream, fused with per-frame gates and a two-layer temporal transformer;
- all **baselines**: metadata random forest, ResNet50-SE-BiLSTM-MHA, Xception, EfficientNet-B0, R(2+1)D-18, VideoMAE-B, and DINOv2 and CLIP (UnivFD) linear probes;
- the **analysis scripts**: statistical tests, robustness, ablation, LODO, resolution control, gate/knock-out, Grad-CAM and cost;
- the **predictions of every reported run** (`paper_pipeline/results/runs/*/preds.csv`), so that all tables and tests can be reproduced without retraining.

## Data set

1,355 videos from seven disaster types (avalanche, earthquake, hurricane, landslide, volcanic eruption, flood, wildfire):

- 628 AI-generated videos, most of them produced by the authors with several text-to-video applications and the rest collected from social media;
- 727 real videos collected from YouTube, Instagram, X and Facebook.

**Download:** [data set link to be added]

Place the videos at the repository root in the following layout (or set the `DISASTER_ROOT` environment variable to the folder that contains `class_dataset/`):

```
class_dataset/
  fake/<Disaster>_Video_Dataset/*.mp4
  real/<Disaster>_Video_Dataset/*.mp4
```

The video IDs used in all CSV files are the first 12 hex digits of the MD5 of the lower-cased relative path `label/<folder>/<file>` (see `paper_pipeline/common.py`). `paper_pipeline/results/splits.csv` gives the class, disaster type, near-duplicate group and cross-validation fold of every video.

## Installation

Tested with Python 3.10, PyTorch 2.8 (CUDA 12.8) and a single NVIDIA RTX 4090 (24 GB).

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

All commands are run from `paper_pipeline/`.

```bash
# 1. Metadata audit (container-level shortcut)
python ../analysis/video_metadata.py
python ../analysis/metadata_shortcut_baseline.py

# 2. Near-duplicate groups and group-aware folds
python find_duplicates.py
python make_groups.py                 # -> results/splits.csv (provided)

# 3. Shortcut-controlled frame cache (16 frame pairs per video)
python prep_cache.py                  # -> cache/clean_pairs/

# 4. Proposed model, group five-fold CV with robustness evaluation
python train.py --model tristream --data clean_pairs --recipe proposed \
  --sem_backbone vit_large_patch14_clip_224.openai --streams sem,tmp \
  --split group --robust --tag Ab_clip_sem_tmp_R

# 5. Same model, leave-one-disaster-out
python train.py --model tristream --data clean_pairs --recipe proposed \
  --sem_backbone vit_large_patch14_clip_224.openai --streams sem,tmp \
  --split lodo --tag L_clip_semtmp

# 6. Linear probes, tables, statistics and figures
python probe_baselines.py
python probe_robust.py
python probe_extra.py
python analyze.py
python paper_tables.py
python review_stats.py
python make_figures.py
python fig_arch.py
```

The full sequence of runs reported in the paper (baselines, ablations, seeds, LODO, resolution harmonization) is given in `paper_pipeline/scripts/run_queue*.sh`. Set `PY` to your Python interpreter before running them, e.g. `PY=python bash scripts/run_queue5.sh`.

### Main scripts

| Script | Purpose |
|---|---|
| `common.py` | paths, video IDs, frame settings |
| `prep_cache.py` | band detection, margin crop, aspect-preserving crop, frame-pair sampling |
| `find_duplicates.py`, `make_groups.py`, `dup_calibrate.py` | near-duplicate detection and group-aware folds |
| `models.py` | proposed network (`tristream`, streams `sem,tmp`) and all neural baselines |
| `train.py` | training and evaluation (group CV, random CV, LODO; `--robust` adds distorted test sets) |
| `probe_baselines.py`, `probe_robust.py`, `probe_extra.py` | DINOv2 / CLIP linear probes |
| `analyze.py`, `paper_tables.py`, `review_stats.py` | McNemar, paired bootstrap, Holm correction, paper tables |
| `res_analysis.py`, `harmonize_cache.py`, `harm_eval.py` | resolution control (matched subset, 288 px harmonization) |
| `explain.py`, `explain_fig.py` | gate weights, stream knock-out, Grad-CAM, error analysis |
| `bench.py` | parameters, GMACs, latency, memory |
| `make_figures.py`, `fig_arch.py` | figures of the paper |

## Results (group five-fold CV, 1,355 videos)

| Model | Accuracy | F1 | JPEG 70 | Down ×0.25 | Blur σ=1 |
|---|---|---|---|---|---|
| CLIP linear probe (UnivFD) | 0.973 | 0.975 | 0.922 | 0.818 | 0.925 |
| **Proposed** | **0.982** | **0.984** | **0.971** | **0.929** | **0.956** |

See the paper for all baselines, statistical tests and analyses.

## Citation

```bibtex
@article{isguzar2026gated,
  title   = {Gated semantic--temporal fusion for detecting AI-generated natural disaster videos on social media under a shortcut-controlled protocol},
  author  = {I{\c{s}}g{\"u}zar, Seda and Polat, Onur and Tonkal, {\"O}zg{\"u}r and Karadeniz, Alper Talha and K{\i}l{\i}{\c{c}}, Kaz{\i}m and C{\"o}mert, Zafer and T{\"u}rko{\u{g}}lu, Muammer},
  journal = {Expert Systems with Applications},
  note    = {Under review},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Real videos remain subject to the terms of their source platforms.
