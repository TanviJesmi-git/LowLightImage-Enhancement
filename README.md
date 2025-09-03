a deep learning based low-light image enhancement system using the Zero-DCE model.The goal is to enhance faintly lit → dark images (e.g. webcam frames at night) while leaving well-lit images unchanged.This module acts as a preprocessing stage for downstream tasks like drowsiness deection .

Approach
Base model: Zero-DCE(Zero-Reference Deep Curve Estimation).
Pretrained weights: Started from the official pretrained Zero-DCE weights.
Fine-tuning:Used a custom face dataset containing both dark/faintly lit and bright images.
Modified training losses to:Boost faintly lit/dark images (exposure-aware loss).Preserve well-lit images (identity loss).
Result: A model that enhances visibility in dark conditions but avoids overexposing already bright faces.

Zero-DCE/
│── Zero-DCE_code/
│   ├── model/                  # Model architecture (enhance_net_nopool.py)
│   ├── train_zero_dce.py       # Training script
│   ├── test_zero_dce.py        # Testing script
│   ├── api_enhancement.py      # FastAPI server (Module 1 API)
│── lowlightDataset/            # Training/validation/test dataset (custom)
│── requirements.txt            # Dependencies
│── Dockerfile                  # Container setup for API
│── README.md                   # This file



