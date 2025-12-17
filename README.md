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
* [cite_start]**Public Leaderboard (NLS):** 0.91345 [cite: 151]
* [cite_start]**Private Leaderboard (NLS):** 0.88077 [cite: 152]

---

## 👥 Team Members

| Name | Affiliation |
| :--- | :--- |
| **Md Nasiat Hasan Fahim** | [cite_start]Dept of CSE, SUST (Session: 2020-21) [cite: 3, 4, 5] |
| **Miftahul Alam Adib** | [cite_start]Dept of Statistics, SUST (Session: 2023-24) [cite: 7, 8, 9] |
| **Arif Hussain** | [cite_start]Dept of Mathematics, SUST (Session: 2022-23) [cite: 11, 12, 13] |

---

## 🧩 Problem Statement

Standard ASR models often fail on regional dialects due to "accent mismatch". Key challenges included:
* [cite_start]**Acoustic Variability:** Phonetic shifts, such as standard `/p/` (*Pani*) becoming `/f/` (*Fani*) in Noakhali/Sylhet[cite: 32, 33].
* [cite_start]**Morphological Variation:** Different verb conjugations (e.g., Standard *Jabo* vs. Regional *Zaiyum* or *Zamu*)[cite: 34, 35].
* [cite_start]**Class Imbalance:** Significant disparity in data availability (e.g., 401 samples for Chittagong vs. 21 for Khulna)[cite: 52].

---

## 🛠 Methodology

### 1. Model Architecture & Initialization
[cite_start]We utilized the **Whisper Medium (769M parameters)** model[cite: 94]. [cite_start]Instead of generic pre-trained weights, we initialized our model using the **1st Place Solution checkpoint from the Bengali.AI Speech Recognition competition**, providing a robust foundation for Bengali acoustics[cite: 96, 101].

### 2. Dual-Stage Sequential Fine-Tuning
To prevent catastrophic forgetting, we employed a two-phase training curriculum with specific hyperparameter schedules:

| Phase | Dataset Mix | Epochs | Warmup Steps | Learning Rate | Composite Score Formula |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Main + DL Sprint | 10 | 100 | 1e-4 | $S_{final} = 0.89 \times WER_{main} + 0.11 \times WER_{diff}$ |
| **Phase 2** | Main + Bengali.AI | 8 | 0 | 1e-4 | $S_{final} = 0.95 \times WER_{main} + 0.05 \times WER_{diff}$ |

* [cite_start]**Adaptive Weighting:** We used composite scoring to balance the learning rate between the main dialect dataset and auxiliary datasets[cite: 170, 176].
* [cite_start]**High-Rank LoRA:** We implemented LoRA with **Rank 1024**, Alpha 64, and Dropout 0.1, targeting `q_proj` and `v_proj` modules to capture long-tail vocabulary[cite: 160].

### 3. Data Preprocessing
* [cite_start]**Audio:** Resampled to 16 kHz mono; generated Log-Mel Spectrograms[cite: 105, 106].
* [cite_start]**Text:** Normalized by removing non-speech artifacts (`<>`, `..`) and English characters[cite: 109].
* [cite_start]**Dynamic Padding:** Custom data collator for batch-level dynamic padding and masking (-100)[cite: 116].

### 4. Post-Processing Pipeline
* [cite_start]**Inference:** Greedy Decoding (`num_beams=1`) with batch size 4 on T4 GPUs to maximize throughput[cite: 181, 182].
* [cite_start]**Repetition Suppression:** Truncated word sequences repeating more than 8 times to remove "stuttering" artifacts[cite: 184].
* **Deep Punctuation Restoration:** An ensemble of four **BERT (MuRIL-base)** models was used. [cite_start]We applied **Class-Weighted Voting [1.0, 1.4, 1.0, 0.8]** to optimize the precision of standard Bengali punctuation (।, ?, ,)[cite: 185].

---

## 📊 Dataset Details

We augmented the primary competition dataset with strictly filtered external resources.

| Dataset | Type | Samples | Filtering Criteria |
| :--- | :--- | :--- | :--- |
| **Shobdotori** | Primary (Dialect) | 3,350 | [cite_start]Stratified sampling to preserve distribution across 20 dialects[cite: 113]. |
| **DL Sprint** | Auxiliary | ~2,389 | **Quality:** >3 upvotes & <1 downvote; [cite_start]**Length:** 4-11 words[cite: 63, 64]. |
| **Bengali.AI** | Auxiliary | ~3,719 | [cite_start]**Conciseness:** 4-5 word phrases only[cite: 60]. |

---

## ⚠️ Challenges & Error Analysis

Despite our robust pipeline, we identified specific linguistic and technical challenges:
* [cite_start]**Phonetic Confusion:** Distinct dialectal sounds (e.g., specific Sylheti tones) were occasionally mapped incorrectly to standard Bengali phonemes[cite: 192].
* [cite_start]**Vocabulary Gaps:** The model struggled with "long-tail" vocabulary absent from standard training corpora[cite: 193].
* [cite_start]**Punctuation Artifacts:** The BERT ensemble sometimes inserted periods prematurely in complex, multi-clause sentences[cite: 195].
* [cite_start]**Hardware Constraints:** Development was constrained by Kaggle’s Tesla T4 (15GB VRAM), necessitating small batch sizes and memory-efficient inference[cite: 197].

---

## 📈 Results

| Experiment Configuration | Public LB (NLS) | Private LB (NLS) |
| :--- | :--- | :--- |
| Baseline (Whisper Small, Static Pad) | 0.76897 | 0.71913 |
| Interim (Whisper Medium, Main Only) | 0.91664 | 0.87203 |
| **Proposed (Dual-Stage + LoRA + Post-Proc)** | **0.91345** | **0.88077** |

---

## 🙏 Acknowledgments

[cite_start]We explicitly thank the **AI-FICATION organizing committee** and the **Department of Electronics & Telecommunication Engineering, CUET**, for organizing this competition[cite: 204, 205].

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
