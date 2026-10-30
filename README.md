# Learning Emotion-Aligned Representations from Audio and Lyrics
Music emotion is expressed through both audio and lyrics. This project investigates whether contrastive audio–lyrics alignment improves music emotion recognition. Using the MERGE Bimodal dataset, it compares unimodal and multimodal models, with and without symmetric contrastive alignment, in terms of emotion prediction and cross-modal retrieval.

## Files
- **Report:** [`Report.pdf`](Report.pdf)
- **Main notebook:** [`notebooks/merge_project.ipynb`](notebooks/merge_project.ipynb) (committed run, all outputs saved)
- **QP@10 check:** [`notebooks/qp10_true_pair_excluded.ipynb`](notebooks/qp10_true_pair_excluded.ipynb)
- **Experimental configuration:** [`experiment_config.txt`](experiment_config.txt)

## Method in brief
- Frozen backbones: **MERT-v1-95M** (audio) and **RoBERTa-base** (lyrics); all 13 hidden states cached in fp16.
- Audio: 30-s clip at 24 kHz, encoded in 5-s windows, pooled to 2 steps/s. Lyrics: full text in overlapping 510-token windows, one vector per non-empty line.
- Per-modality lightweight Transformer encoder (learnable layer mix → 768→256 projection → [CLS] + positions → 2 pre-LN layers), giving uᵃ and uˡ.
- Fusion: joint self-attention over both full token sequences with modality embeddings and a fusion [CLS], giving uᵐ. Heads give qᵃ, qˡ, qᵐ.
- Loss: L = L_emo + λ_align · L_align, where L_emo = CE(qᵐ) + λ_uni/2 [CE(qᵃ) + CE(qˡ)] and L_align is symmetric InfoNCE on L2-normalised uᵃ, uˡ with fixed τ.

## Experiments
- **E1:** audio only (MERT-v1-95M)
- **E2:** lyrics only (RoBERTa-base)
- **E3:** multimodal, without alignment
- **E4:** multimodal, with contrastive alignment
Hyperparameters were chosen on validation with seeds 101–103. The selected settings were retrained with seeds 0–4 and evaluated once on test.

## Main results (test set, mean ± SD over 5 seeds)
| | Macro-F1 | MRR | Median rank | QP@10 |
|---|---|---|---|---|
| E1 audio | 0.683 ± 0.012 | – | – | – |
| E2 lyrics | 0.731 ± 0.019 | – | – | – |
| E3 multimodal | 0.826 ± 0.014 | 0.025 ± 0.006 | 131 ± 30 | 0.39 ± 0.07 |
| E4 multimodal + alignment | 0.818 ± 0.017 | 0.051 ± 0.002 | 63 ± 5 | 0.59 ± 0.02 |

Fusion clearly improved prediction. Alignment improved audio–lyrics retrieval (MRR 0.025 → 0.051) but did not improve emotion prediction.

## How to run
1. Open `notebooks/merge-project.ipynb` on Kaggle with a **T4 GPU**.
2. Add the MERGE Bimodal Complete dataset as input.
3. Click **Run All**.

## Data
Louro et al., 2024 ([arXiv:2407.06060](https://arxiv.org/abs/2407.06060)).
