# Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach

**Team Name:** Backprop Sust  
**Competition:** AI-FICATION 2025 - Shobdotori Challenge

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)](https://pytorch.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-orange)](https://huggingface.co/docs/peft/index)

## 📌 Project Overview

This repository contains the solution developed by **Team Backprop Sust** for the Shobdotori ASR challenge. The objective was to develop a robust Automatic Speech Recognition (ASR) system capable of transcribing **20 distinct regional Bangladeshi dialects** (e.g., Chittagonian, Sylheti, Rangpuri) into **Standard Formal Bangla** text.

Our solution leverages the **OpenAI Whisper Medium** architecture, optimized via a novel **Dual-Stage Sequential Fine-Tuning** strategy to handle acoustic variability and data scarcity.

### 🏆 Key Achievements
* **Public Leaderboard (NLS):** 0.91345
* **Private Leaderboard (NLS):** 0.88077

---

## 👥 Team Members

| Name | Affiliation |
| :--- | :--- |
| **Md Nasiat Hasan Fahim** | Dept of CSE, SUST (Session: 2020-21) |
| **Miftahul Alam Adib** | Dept of Statistics, SUST (Session: 2023-24) |
| **Arif Hussain** | Dept of Mathematics, SUST (Session: 2022-23) |

---

## 🧩 Problem Statement

Standard ASR models often fail on regional dialects due to "accent mismatch". Key challenges included:
* **Acoustic Variability:** Phonetic shifts, such as standard `/p/` (*Pani*) becoming `/f/` (*Fani*) in Noakhali/Sylhet.
* **Morphological Variation:** Different verb conjugations (e.g., Standard *Jabo* vs. Regional *Zaiyum* or *Zamu*).
* **Class Imbalance:** Significant disparity in data availability (e.g., 401 samples for Chittagong vs. 21 for Khulna).

---

## 🛠 Methodology

### 1. Model Architecture & Initialization
We utilized the **Whisper Medium (769M parameters)** model. Instead of generic pre-trained weights, we initialized our model using the **1st Place Solution checkpoint from the Bengali.AI Speech Recognition competition**, providing a robust foundation for Bengali acoustics.

### 2. Dual-Stage Sequential Fine-Tuning
To prevent catastrophic forgetting, we employed a two-phase training curriculum:

| Phase | Dataset Composition | Strategy | Weighting (Main/Diff) |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Main (Shobdotori) + DL Sprint | Base Adaptation | 0.89 / 0.11 |
| **Phase 2** | Main (Shobdotori) + Bengali.AI Speech | Targeted Refinement | 0.95 / 0.05 |

* **Adaptive Weighting:** We used composite scoring to balance the learning rate between the main dialect dataset and auxiliary datasets.
* **High-Rank LoRA:** We implemented LoRA with **Rank 1024**, Alpha 64, and Dropout 0.1, targeting `q_proj` and `v_proj` modules to capture long-tail vocabulary.

### 3. Data Preprocessing
* **Audio:** Resampled to 16 kHz mono; generated Log-Mel Spectrograms.
* **Text:** Normalized by removing non-speech artifacts (`<>`, `..`) and English characters.
* **Dynamic Padding:** Custom data collator for batch-level dynamic padding.

### 4. Post-Processing Pipeline
* **Inference:** Greedy Decoding (`num_beams=1`) with batch size 4 on T4 GPUs.
* **Repetition Suppression:** Truncated word sequences repeating more than 8 times to remove "stuttering" artifacts.
* **Deep Punctuation Restoration:** An ensemble of four **BERT (MuRIL-base)** models was used to restore punctuation (।, ?, ,) using class-weighted voting.

---

## 📊 Dataset Details

We augmented the primary competition dataset with external resources.

| Dataset | Type | Samples | Filtering Criteria |
| :--- | :--- | :--- | :--- |
| **Shobdotori** | Primary (Dialect) | 3,350 | Stratified Split |
| **DL Sprint** | Auxiliary | ~2,389 | Length 4-11 words, High Upvotes |
| **Bengali.AI** | Auxiliary | ~3,719 | 4-5 word concise phrases |

---

## 📈 Results

| Experiment Configuration | Public LB (NLS) | Private LB (NLS) |
| :--- | :--- | :--- |
| Baseline (Whisper Small, Static Pad) | 0.76897 | 0.71913 |
| Interim (Whisper Medium, Main Only) | 0.91664 | 0.87203 |
| **Proposed (Dual-Stage + LoRA + Post-Proc)** | **0.91345** | **0.88077** |

---

## 📜 Citation

If you find this approach useful, please cite our work:
```bibtex
@inproceedings{backpropsust2025,
  title={Transcribing Regional Bangladeshi Dialects: A Dual-Stage Sequential Fine-Tuning Approach},
  author={Fahim, Md Nasiat Hasan and Adib, Miftahul Alam and Hussain, Arif},
  booktitle={AI-FICATION 2025: Shobdotori Challenge},
  year={2025},
  organization={Chittagong University of Engineering & Technology (CUET)},
  note={Team Backprop Sust, Shahjalal University of Science and Technology}
}
```

