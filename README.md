Multi-Class Classification of Reddit Posts for Mental Health Detection

Overview
This project investigates how well AI can identify signs of mental health conditions from social media text. We fine-tuned three transformer-based models on Reddit posts to classify them into five mental health categories, comparing domain-specific and general-purpose language models.
Research Question: How well can AI help identify signs of mental health conditions from social media text?

Dataset

Source: Reddit Mental Health Data (Kaggle)
Original size: 5,957 samples across 5 classes
Final size: 4,375 samples (after quality filtering)

Preprocessing Pipeline
StepDetailsToken length filteringRemoved posts >512 tokens: -384 samples (6.5%)Duplicate removalRemoved 1,177 duplicatesLabel conflict resolutionRemoved 9 conflicting labels (same text, different labels)Total reduction26.6% (1,570 samples removed)
Final Dataset Split
SplitSizeTrain (70%)3,061 samplesValidation (15%)657 samplesTest (15%)657 samples
Class distribution is balanced at 18.7–21.1% per class (±2.4% variance), with stratified splits ensuring balance across all sets.
Classes

None / Stress
Depression
Bipolar Disorder
Personality Disorder / PTSD
Anxiety


Models
Model 1: MentalBERT
Domain-specific BERT model pre-trained on mental health text.
Best configuration:

Learning Rate: 1e-5, Batch Size: 8, Epochs: 15
Weight Decay: 0.01, Warmup Ratio: 0.1
Freeze Embeddings: True, Freeze 9 layers (3 trainable)
Loss Function: Cross-Entropy, Early Stopping: 2 epochs

Best Macro F1: 79.2%
Key confusion patterns: Depression vs Personality Disorder (11.2–12.3%), Bipolar to Depression (10.5%), Anxiety to None (8.0%).

Model 2: DeBERTa-v3 Base
Tested in two versions based on token length handling.
Version A (posts <512 tokens): Froze first 3 layers, 10 trainable — Best Macro F1: 81.1%
Version B (posts >512 tokens): Froze 2 layers, 10 trainable — handles longer posts with reduced confusion in Personality Disorder class.

Model 3: RoBERTa-base ✅ (code available)
Pre-trained on 160GB corpus including OpenWebText (38GB Reddit data). Full-layer fine-tuning optimized for social media text.
Best configuration:

Learning Rate: 1e-5, Batch Size: 8, Epochs: 16
Label Smoothing: 0.03
Inverse Frequency Class Weights
Full fine-tuning of all RoBERTa layers

Best training point: ~Epoch 2 (mild overfitting observed after)

Results Summary
ModelMacro F1NotesMentalBERT79.2%Fails all class-wise comparisons vs othersDeBERTa-v3 (Version A)81.1%Best on Personality Disorder classRoBERTa-base~82%Best overall; wins 4 out of 5 classes
Key Insights

RoBERTa outperforms MentalBERT despite both being trained on Reddit data — diverse pre-training corpus (160GB) outweighs domain specificity
RoBERTa's optimized architecture (2019) outperforms MentalBERT's base BERT architecture (2018)
DeBERTa wins on Personality Disorder (1/5 classes) due to disentangled attention
All models rank Depression as the hardest category to classify


Repository Structure
mental-health-nlp/
├── models/
│   ├── roberta/
│   │   └── nlp_project_final_12_2.py   ✅ Available
│   ├── mentalbert/                      🔄 Coming soon
│   └── deberta/                         🔄 Coming soon
├── preprocessing/                       🔄 Coming soon
├── data/                                (not included — download from Kaggle)
└── results/                             🔄 Coming soon
