# 🛰️ Vanilla Drone Detection — Development & Research Plan

<p align="center">
  <img src="https://img.shields.io/badge/Task-ISLab%20Pusan%20National%20University-blue?style=for-the-badge&logo=googlescholar" alt="Task" />
  <img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Architecture-Anchor--Free%20FPN%2FPAN%2FCBAM-green?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/Weights-100%25%20From--Scratch-purple?style=for-the-badge" alt="Scratch" />
</p>

---

## 👨‍💻 Project Overview

This repository hosts the from-scratch prototype and experimental framework for the **ISLab Pusan National University AI Engineering Researcher Position Evaluation**.

The objective is to design, implement, and benchmark a **100% Vanilla Anchor-Free Object Detector** for drone detection and classification under severe scale disparity (tiny distant objects) and complex aerial background clutter, without relying on any pretrained weights or third-party detection frameworks (e.g., MMDetection, Detectron2, Ultralytics).

---

## 🎯 Core Requirements & Constraints

- 🚫 **No Pretrained Weights**: Entire network initialized randomly from scratch ($\mathcal{N}(0, \sigma^2)$).
- 📐 **Custom Loss Function**: Hand-crafted multi-task objective addressing extreme aerial scale disparity and class imbalance.
- 🔬 **Systematic Ablation Study**: Multi-tier architectural evolution isolating feature fusion, spatial attention, and stride mechanics.
- 📄 **IEEE Conference Paper**: 3–4 page manuscript documenting methodology, mathematical formulations, and comparative findings.
- 🐳 **Clean Code & Containerization**: OOP design with clean modular separation and Docker orchestration.

---

## 🗺️ Planned Development Roadmap

```mermaid
graph TD
    A[Phase 1: Environment & Project Setup] --> B[Phase 2: Data Pipeline & Split Strategy]
    B --> C[Phase 3: Custom Multi-Task Losses & Box Ops]
    C --> D[Phase 4: Backbone & Decoupled Detection Heads]
    D --> E[Phase 5: Multi-Tier Ablation Architecture Design]
    E --> F[Phase 6: AMP Training Engine & Assigner]
    F --> G[Phase 7: Baseline Benchmarks & Memory Tuning]
    G --> H[Phase 8: High-Res Model-D & Model EMA]
    H --> I[Phase 9: Kaggle Pipeline & Final Submission]
```

### 1. Data Pipeline & Temporal Leakage Prevention
- **`DroneYOLODataset`**: Pure PyTorch dataset loader with high-speed in-memory pre-resized caching for maximum GPU throughput.
- **Leakage-Free Splitting**: Sequence-aware regex grouping (`scripts/prepare_splits.py`) ensuring frames from the same video sequence are never split across train and validation sets.
- **Augmentation Suite**: Multi-scale training, photometric color jitter, and random horizontal flipping.

### 2. Custom Multi-Task Objective ($\mathcal{L}_{\text{hybrid}}$)
- **Focal Loss ($\mathcal{L}_{\text{cls}}$)**: Suppresses overwhelming background easy negatives ($>99.8\%$ of spatial cells).
- **Normalized Wasserstein Distance ($\mathcal{L}_{\text{NWD}}$)**: Models boxes as 2D Gaussian distributions for smooth geometric gradients on tiny sub-20px drones.
- **Complete IoU ($\mathcal{L}_{\text{CIoU}}$)**: Enforces scale-invariant overlap, Euclidean center distance, and aspect ratio alignment.
- **Centerness / Objectness Quality Loss ($\mathcal{L}_{\text{obj}}$)**: Mitigates low-quality boundary false positives.

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{reg}} \left( \alpha \mathcal{L}_{\text{CIoU}} + (1 - \alpha) \mathcal{L}_{\text{NWD}} \right)$$

### 3. Progressive Architectural Ablation Suite
To provide empirical attribution for every design component, four models will be developed and benchmarked:

| Model | Neck Architecture | Attention Mechanism | Feature Strides | Focus Area |
| :--- | :--- | :--- | :--- | :--- |
| **Model-A** (`model_a_fpn`) | Top-Down FPN | None | $[8, 16, 32]$ | Baseline top-down semantic pyramid |
| **Model-B** (`model_b_pan`) | Bidirectional FPN + PAN | None | $[8, 16, 32]$ | Bottom-up spatial localization cues |
| **Model-C** (`model_c_attn`) | Bidirectional FPN + PAN | CBAM (Channel + Spatial) | $[8, 16, 32]$ | Saliency focus & aerial clutter suppression |
| **Model-D** (`model_d_p2_ema`) | 4-Level High-Res FPNPAN4 | CBAM + Model EMA | $[4, 8, 16, 32]$ | High-resolution P2 stride-4 for sub-20px drones |

### 4. Training Engine & Cloud Benchmarking
- Mixed-precision (`torch.cuda.amp`) training loop with gradient clipping and cosine annealing learning rate scheduler.
- Scale-aware `TargetAssigner` with center radius sampling.
- Standardized evaluation measuring COCO-style $\text{mAP}@50$, $\text{mAP}@50:95$, Precision, Recall, inference latency (ms), and FPS throughput.
- Comparative baseline benchmarks against standard reference models: **YOLOv8-nano (3.2M)**, **YOLOv8-small (11.2M)**, and **RT-DETR-L (32.0M)**.

---

## 🛠️ Tech Stack & Tools

<table>
  <tr>
    <td><strong>Core Framework</strong></td>
    <td>
      <img src="https://skillicons.dev/icons?i=python,pytorch" /><br/>
      <sub>PyTorch 2.2+ · Torchvision · PyYAML</sub>
    </td>
  </tr>
  <tr>
    <td><strong>Computer Vision</strong></td>
    <td>
      <img src="https://skillicons.dev/icons?i=opencv" /><br/>
      <sub>OpenCV · PIL · NumPy · Matplotlib</sub>
    </td>
  </tr>
  <tr>
    <td><strong>MLOps & Cloud</strong></td>
    <td>
      <img src="https://skillicons.dev/icons?i=docker,git,github" /><br/>
      <sub>Docker · Docker Compose · Kaggle GPU (T4/P100) · Weights & Biases</sub>
    </td>
  </tr>
  <tr>
    <td><strong>Documentation</strong></td>
    <td>
      <img src="https://skillicons.dev/icons?i=latex" /><br/>
      <sub>IEEE Conference LaTeX Template · Overleaf</sub>
    </td>
  </tr>
</table>

---

## 👤 Author & Contact

**Fajar Wira Adikusuma**  
*Data Scientist / AI Engineer*  

<p align="left">
  <a href="https://github.com/aeoncyr"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-aeoncyr-181717?style=flat-square&logo=github"></a>
  <a href="https://www.linkedin.com/in/fajarwiraa/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-fajarwiraa-0A66C2?style=flat-square&logo=linkedin"></a>
  <a href="mailto:fajar.wira.a@gmail.com"><img alt="Email" src="https://img.shields.io/badge/Email-fajar.wira.a%40gmail.com-EA4335?style=flat-square&logo=gmail"></a>
</p>
