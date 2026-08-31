# 🚀 Kaggle Training Guide for Vanilla Drone Detection

Since my laptop is a potato (AMD Ryzen 5 integrated graphics), training deep learning models locally will be CPU-bound (~25 min/epoch). Kaggle provides **30 hours/week of free NVIDIA GPU access** (T4 x2 or P100), where 50 epochs will finish in only **~10–15 minutes**.

---

## 📋 Step-by-Step Workflow

### Step 1: Upload Dataset to Kaggle
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **"New Dataset"**.
2. Upload the `datasets/obj_det_base` folder (or use the compressed dataset.zip` and upload).
3. Title it: `drone-detection-dataset` (set visibility to **Private**).
4. Click **Create**.

---

### Step 2: Create a New Kaggle Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. Click **File** ➔ **Upload Notebook** and choose:
   `kaggle/drone_detector_kaggle.ipynb`
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   - **Internet**: Toggle to **ON** (needed for W&B & ultralytics baselines).
   - **Language**: Python.

---

### Step 3: Attach the Dataset
1. In the right-hand panel, click **+ Add Input** / **Add Data**.
2. Select **Your Datasets** and add `drone-detection-dataset`.
3. The dataset will be mounted read-only at `/kaggle/input/drone-detection-dataset/...`.

---

### Step 4: Run the Notebook Cells

| Cell | Purpose | Expected Output / Time |
|---|---|---|
| **1. GPU Check** | Confirms CUDA and GPU device (Tesla T4 / P100) | `CUDA Available: True` |
| **2. Dependencies** | Installs `wandb`, `ultralytics`, `pyyaml` | ~15 seconds |
| **3. Bootstrap Code** | Writes the modular `src/` package to `/kaggle/working/` | `[✓] Source modules loaded` |
| **4. Split Data** | Auto-detects dataset and generates sequence-level 80/20 split | `Train: ~1898, Val: ~502` |
| **5. Config Builder** | Defines hyperparameters & zipping helper for Model-A, B, C, D | Instant |
| **5.5 Restore Runs** | *(Optional)* Auto-restores uploaded `model_*_artifacts.zip` or runs | Instant checkpoint scan |
| **6. Train Model-A** | Trains FPN baseline (`RUN_MODEL_A = True`) + packages zip | ~10 mins for 50 epochs |
| **7. Train Model-B** | Trains FPN+PAN variant (`RUN_MODEL_B = True`) + packages zip | ~10 mins for 50 epochs |
| **8. Train Model-C** | Trains FPN+PAN+CBAM (`RUN_MODEL_C = True`) + packages zip | ~11 mins for 50 epochs |
| **9. Train Model-D** | Trains 4-Level P2-P5+EMA (`RUN_MODEL_D = True`) + packages zip | ~12 mins for 50 epochs |
| **10. Plot Curves** | Autonomous multi-model comparative ablation & loss curves | Instant visual charts |
| **11. Quantitative Eval**| Evaluates all available checkpoints & detects top model | Exact mAP, latency, & FPS |
| **12. Detections** | Visualizes predicted bounding boxes on validation images | Saves `detection_samples.png` |
| **13. Baselines** | Trains YOLOv8-nano, YOLOv8-small, and RT-DETR-L baselines | ~20 mins total |
| **14. Master Comparison**| Generates comparative table (mAP, latency, FPS) & publication charts | Saves CSV, MD, & PNG |
| **15. Export Summary Zip**| Bundles summary tables, charts, and metrics into `master_comparison_artifacts.zip` | 1-click download |

---

## 📥 Downloading Results to Local

Once each step finishes:
1. Immediately download individual model checkpoints via their 1-click links:
   - `model_a_artifacts.zip`
   - `model_b_artifacts.zip`
   - `model_c_artifacts.zip`
   - `model_d_artifacts.zip`
2. Download the final `master_comparison_artifacts.zip` containing:
   - `model_comparison_table.csv` & `model_comparison_table.md`
   - `model_comparison_chart.png` (mAP & latency comparison for the paper)
   - `ablation_curves.png` (ablation loss & mAP curves for the paper)
   - `detection_samples.png` (qualitative detection figure for the paper)
   - `splits/train.txt` & `splits/val.txt` (reproducible data split)
3. Unzip into local project directory:
   ```bash
   # Extract into local repo to run local evaluations/infer
   unzip drone_detection_submission_artifacts.zip -d .
   ```
