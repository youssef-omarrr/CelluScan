# [🧫 CelluScan: Automated Blood Cell Classification with Vision Transformer](https://huggingface.co/spaces/Youssef-omarr/CelluScan)

CelluScan is a custom Vision Transformer (ViT)–based pipeline for classifying white blood cell (WBC) subtypes and detecting abnormalities. Trained on \~60,000 curated images, the model achieves 88.93% test accuracy and is deployed as a service.

![alt text](image.png)

---

### Repository Structure

```
CelluScan/
├── Blood cells datasets/        # Folders of all classes (with one image as an example)
│
├── Extra info/                  # Supplementary research and technical notes
│   ├── MultiHead_Self_Attention.md   # Notes on attention mechanisms
│   └── Researches.md                # Literature and research summaries on the blood cells
│
├── Outputs/                     # Model outputs, confusion matrices, and logs
│   ├── Cmatrix_after_7_EPOCHS.png         # Confusion matrix after 7 epochs
│   ├── Cmatrix_after_another_5_EPOCHS.png # Confusion matrix after additional epochs
│   ├── losses_after_7_EPOCHS.png          # Loss curves after 7 epochs
│   └── losses_after_another_5_EPOCHS.png  # Loss curves after additional epochs
│
├── AN IMAGE IS WORTH 16X16.pdf  # Reference paper on Vision Transformers
├── CelluScan.ipynb              # Main Jupyter notebook for model training and evaluation
└── final_fine_tuning.ipynb      # Notebook for final model fine-tuning
```

---

### Table of Contents

1. [Overview](#overview)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training Procedure](#training-procedure)
5. [Results & Discussion](#results--discussion)
6. [License](#license)

---

## Overview

* **Goal**: Classify 14 WBC categories (e.g., Neutrophils, Lymphocytes) and identify developmental stages (e.g., Promyelocyte, Myelocyte).
* **Approach**: Transfer learning using PyTorch’s pretrained ViT-B/16, extended with custom classification heads and attention modules. Scratch training remains available but is resource-intensive.

> **Note:** On the [Hugging Face website](https://huggingface.co/spaces/Youssef-omarr/CelluScan), you can also view potential diagnoses and interesting facts about each cell type alongside the model predictions.

## Dataset & Preprocessing

* **Total**: \~60,000 images across 14 classes.
* **Initial Distribution**: Ranged from 151 (Immature Granulocyte) to 8,685 (Lymphocyte).
* **Balancing**: Minority classes augmented with `TrivialAugmentWide` to ≥4,500 images each.
* **Splits**: 5% (sanity checks), 20% (hyperparameter tuning), 100% (final training).

| Class                | Original | Post-Balance |
| -------------------- | -------: | -----------: |
| Immature Granulocyte |      151 |        4,500 |
| Promyelocyte         |      592 |        4,500 |
| Myeloblast           |    1,000 |        4,500 |
| Metamyelocyte        |    1,015 |        4,500 |
| Myelocyte            |    1,137 |        4,500 |
| Erythroblast         |    1,551 |        4,500 |
| Band Neutrophil      |    1,634 |        4,500 |
| Basophil             |    1,653 |        4,500 |
| Platelet             |    2,348 |        4,500 |
| Segmented Neutrophil |    2,646 |        4,500 |
| Monocyte             |    5,046 |        5,046 |
| Neutrophil           |    6,779 |        6,779 |
| Eosinophil           |    7,141 |        7,141 |
| Lymphocyte           |    8,685 |        8,685 |

## Model Architecture

> **Note:** We leverage PyTorch’s pretrained ViT-B/16 for efficient transfer learning. Custom modules wrap and extend this core model; full-from-scratch training is available if needed.

* **Patch Embedding**: 224×224 images → 16×16 patches → 768-dim embeddings.
* **Transformer Blocks**: 12 layers, 12 heads (64-dim), MLP (2,048-dim, GELU).
* **Classification**: Learnable \[CLS] token, final MLP head.
* **Implementation**: Custom PyTorch modules (`PatchEmbedding`, `MultiHeadSelfAttention`, `TransformerBlock`, `ViTClassifier`).

## Training Procedure

1. **Data Splits**:

   * 5% (\~3K images, 5 epochs)
   * 20% (\~12K images, 7 epochs)
   * 100% (\~60K images, 7 epochs)
2. **Optimizer**: `torch.optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=1e-2)`
3. **Loss**: `torch.nn.CrossEntropyLoss()`
4. **Metrics**: `torchmetrics.Accuracy(task="multiclass", num_classes=14)`


## Results & Discussion

![alt text](Outputs/Cmatrix_after_another_5_EPOCHS.png)

* **Overall Test Accuracy**: 88.93% on full dataset.
* **Class-wise Challenges**:

  * **Metamyelocytes vs. Myelocytes vs. Promyelocytes**: High visual similarity leads to lower recall (\~75%).
  * **Band vs. Segmented Neutrophils**: Maturity boundaries are subtle; recall \~78%.

Despite these challenges, the model demonstrates robust performance for automated screening; further chemical staining data could bolster clinically critical distinctions.


## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
