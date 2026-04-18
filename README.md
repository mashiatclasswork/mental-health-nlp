# Multi-Class Classification of Reddit Posts for Mental Health Detection

---

## Overview

This project investigates how well AI can identify signs of mental health conditions from social media text. We fine-tuned three transformer-based models on Reddit posts to classify them into five mental health categories, comparing domain-specific and general-purpose language models.

**Research Question:** How well can AI help identify signs of mental health conditions from social media text?

---

## Dataset

- **Source:** Reddit Mental Health Data (Kaggle)
- **Original size:** 5,957 samples across 5 classes
- **Final size:** 4,375 samples after quality filtering

### Preprocessing Pipeline

| Step | Details |
|------|---------|
| Token length filtering | Removed posts >512 tokens: 384 samples (6.5%) |
| Duplicate removal | Removed 1,177 duplicates |
| Label conflict resolution | Removed 9 conflicting labels (same text, different labels) |
| Total reduction | 26.6% (1,570 samples removed) |

### Dataset Split

| Split | Size |
|-------|------|
| Train (70%) | 3,061 samples |
| Validation (15%) | 657 samples |
| Test (15%) | 657 samples |

Class distribution is balanced at 18.7–21.1% per class (±2.4% variance) with stratified splits across all sets.

### Classes
1. None / Stress
2. Depression
3. Bipolar Disorder
4. Personality Disorder / PTSD
5. Anxiety

---

## Models

### Model 1: MentalBERT

**Best configuration:** Learning Rate 1e-5, Batch Size 8, Epochs 15, Weight Decay 0.01, Warmup Ratio 0.1, Freeze 9 layers (3 trainable), Cross-Entropy loss, Early Stopping after 2 epochs.

**Best Macro F1: 79.2%**

Key confusion patterns: Depression vs Personality Disorder (11.2–12.3%), Bipolar to Depression (10.5%), Anxiety to None (8.0%).

---

### Model 2: DeBERTa-v3 Base

Tested in two versions based on token length.

- **Version A (posts under 512 tokens):** Froze first 3 layers, 10 trainable. **Best Macro F1: 81.1%**
- **Version B (posts over 512 tokens):** Froze 2 layers, 10 trainable. Reduced confusion in Personality Disorder class.

---

### Model 3: RoBERTa-base (code available)

Pre-trained on 160GB corpus including OpenWebText (38GB Reddit data). Full-layer fine-tuning optimized for social media text.

**Best configuration:** Learning Rate 1e-5, Batch Size 8, Epochs 16, Label Smoothing 0.03, Inverse Frequency Class Weights, full fine-tuning of all layers.

Best training point around Epoch 2, with mild overfitting observed after.

---

## Results Summary

| Model | Macro F1 | Notes |
|-------|----------|-------|
| MentalBERT | 79.2% | Weakest overall across all class comparisons |
| DeBERTa-v3 | 81.1% | Best on Personality Disorder class |
| RoBERTa-base | ~82% | Best overall, wins 4 out of 5 classes |

### Key Insights

- RoBERTa outperforms MentalBERT despite both being Reddit-trained — diverse pre-training corpus outweighs domain specificity
- RoBERTa's optimized 2019 architecture outperforms MentalBERT's base BERT (2018)
- DeBERTa wins on Personality Disorder due to disentangled attention mechanism
- All three models rank Depression as the hardest category to classify

---

## Repository Structure

```
mental-health-nlp/
├── models/
│   ├── roberta/
│   │   └── nlp_project_final_12_2.py   (available)
│   ├── mentalbert/                      (coming soon)
│   └── deberta/                         (coming soon)
├── preprocessing/                       (coming soon)
├── data/                                (not included, download from Kaggle)
└── results/                             (coming soon)
```

---

## Requirements

```bash
pip install transformers torch scikit-learn pandas numpy
```

---

## References

- Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
- He & Garcia (2009). Learning from imbalanced data.
- Japkowicz & Stephen (2002). The class imbalance problem.

---

