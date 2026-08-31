import json
import base64
import io
import zipfile
from pathlib import Path

def build_notebook():
    cells = []

    def md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header
    md("""
# 🛰️ Vanilla Drone Detection — Kaggle Training & Benchmark Notebook
### Scholarship in AI Engineering Task
- **Core Requirement:** 100% Vanilla from-scratch Anchor-Free Object Detector (No pretrained weights).
- **Custom Loss:** Focal Loss + Hybrid CIoU/NWD Multi-Task Objective.
- **Architectures:** Model-A (FPN), Model-B (FPN+PAN), Model-C (FPN+PAN+CBAM), Model-D (P2-P5 4-Level High-Res + CBAM + EMA).
- **Baselines:** YOLOv8-nano, YOLOv8-small, and RT-DETR-L comparison.
- **Modularity:** Every model training cell is independent with its own execution toggle and 1-click artifact exporter.
""")

    # Step 1: GPU & Environment Setup
    md("## 1. Environment & GPU Verification")
    code("""!nvidia-smi
import torch
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count    : {torch.cuda.device_count()}")
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
""")

    # Step 2: Install additional packages if needed
    md("## 2. Dependencies Setup (W&B, Ultralytics for baselines)")
    code("""!pip install -q wandb ultralytics pyyaml
import os, sys, glob, gc, shutil, zipfile
import yaml
from pathlib import Path
""")

    # Step 3: Project Code Setup (Self-contained in-cell zip unpacker)
    root = Path(__file__).resolve().parent.parent
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in ["src", "configs", "train.py", "evaluate.py", "infer.py", "test_pipeline.py"]:
            item_path = root / item
            if item_path.is_file():
                zf.write(item_path, item)
            elif item_path.is_dir():
                for p in sorted(item_path.rglob("*")):
                    if p.is_file() and not p.name.endswith(".pyc") and "__pycache__" not in p.parts:
                        rel = p.relative_to(root).as_posix()
                        zf.write(p, rel)

    b64_zip = base64.b64encode(zip_buf.getvalue()).decode("ascii")

    md("""## 3. Bootstrap Vanilla Drone Detector Codebase
This cell automatically unpacks and overwrites the entire modular `src/` package directly into `/kaggle/working/src/` from embedded base64 assets.
- **Rerun Safe**: Automatically clears Python module caches and purges stale bytecode so code revisions apply immediately without restarting the kernel.""")
    code(f"""import os, sys, io, base64, zipfile, shutil, glob
ROOT = '/kaggle/working'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 1. Purge in-memory module cache to force fresh imports upon rerun
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith('src.') or mod_name == 'src':
        del sys.modules[mod_name]

# 2. Overwrite existing files cleanly from base64 payload
ZIP_PAYLOAD = \"\"\"{b64_zip}\"\"\"

print("[*] Unpacking and overwriting codebase into /kaggle/working ...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP_PAYLOAD.encode('ascii')))) as zf:
    for member in zf.infolist():
        target_path = os.path.join(ROOT, member.filename)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if not member.is_dir():
            with zf.open(member) as source, open(target_path, "wb") as target:
                shutil.copyfileobj(source, target)

# 3. Clean any stale bytecode caches
for pycache in glob.glob(f"{{ROOT}}/src/**/__pycache__", recursive=True):
    shutil.rmtree(pycache, ignore_errors=True)

# 4. Fresh module import verification
from src.models.detector import VanillaDroneDetector
from src.losses.multi_task_losses import HybridMultiTaskLoss
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.utils.ema import ModelEMA
print("[✓] Source modules freshly overwritten, reloaded, and verified successfully!")
""")

    # Step 4: Dataset Detection & Sequence-Level Stratified Split
    md("""## 4. Dataset Discovery & Stratified Sequence-Level Split
Automatically discovers images across all directories in `/kaggle/input/` and creates stratified sequence-level splits to avoid video frame temporal leakage.""")
    code(r"""import re, random
from collections import defaultdict

# 1. Discover all candidate images under /kaggle/input recursively
candidate_files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
    candidate_files.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))

if not candidate_files:
    candidate_files = glob.glob("datasets/obj_det_base/*.png") + glob.glob("/kaggle/working/datasets/obj_det_base/*.png")

print(f"[*] Discovered {len(candidate_files)} image files.")

if len(candidate_files) == 0:
    print("[!] No images found! Checking /kaggle/input structure:")
    for root_dir, dirs, files in os.walk("/kaggle/input"):
        print(f"    Folder: {root_dir} -> {len(files)} files, {len(dirs)} subdirs")
    raise RuntimeError("No image files found in /kaggle/input! Please click '+ Add Input' in Kaggle to attach the dataset.")

# 2. Match with corresponding label (.txt) files
labeled_images = []
for img in candidate_files:
    lbl = Path(img).with_suffix(".txt")
    if lbl.exists():
        labeled_images.append(str(Path(img).resolve()))

if labeled_images:
    images_to_split = labeled_images
    print(f"[*] Found {len(labeled_images)} verified (image + label .txt) pairs.")
else:
    images_to_split = [str(Path(p).resolve()) for p in candidate_files]
    print(f"[*] Using {len(images_to_split)} images without separate label check.")

# 3. Sequence pattern regex for leakage-free video sequence grouping
_FILENAME_PATTERN = re.compile(
    r"(?:augmented_)?raw_dataset_"
    r"(?P<scene>[a-z]+_[a-z]+)"
    r"_[a-z]+_[a-z]+_"
    r"(?P<seq_a>\d+)_(?P<seq_b>\d+)_"
    r"sequence\.\d+_step0\.camera\.(png|jpg|jpeg)$"
)

groups = defaultdict(list)
for img_path in images_to_split:
    fname = Path(img_path).name
    m = _FILENAME_PATTERN.match(fname)
    if m:
        scene = m.group("scene")
        seq_id = f"{m.group('seq_a')}_{m.group('seq_b')}"
        groups[(scene, seq_id)].append(img_path)
    else:
        # Fallback grouping for general / renamed files
        groups[("general", Path(img_path).stem)].append(img_path)

print(f"[*] Grouped into {len(groups)} sequence/image blocks.")

# 4. Stratified 80/20 train/val split
rng = random.Random(42)
scene_to_seqs = defaultdict(list)
for scene, seq_id in groups.keys():
    scene_to_seqs[scene].append((scene, seq_id))

train_paths, val_paths = [], []
for scene, seq_keys in scene_to_seqs.items():
    rng.shuffle(seq_keys)
    n_val = max(1, round(len(seq_keys) * 0.2))
    val_keys = seq_keys[:n_val]
    train_keys = seq_keys[n_val:]
    for k in train_keys: train_paths.extend(groups[k])
    for k in val_keys: val_paths.extend(groups[k])

if len(train_paths) == 0 and len(val_paths) > 1:
    train_paths = val_paths[:-1]
    val_paths = val_paths[-1:]

splits_dir = Path("/kaggle/working/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

with open(splits_dir / "train.txt", "w") as f:
    for p in sorted(train_paths): f.write(p + "\n")

with open(splits_dir / "val.txt", "w") as f:
    for p in sorted(val_paths): f.write(p + "\n")

print(f"[✓] Split saved: Train = {len(train_paths)} images, Val = {len(val_paths)} images")
""")

    # Step 5: Training Configurations Helper
    md("## 5. Experiment Config Generator & Run Packaging Utility")
    code("""def get_config(model_variant="model_a_fpn", epochs=50, batch_size=16, lr=0.001):
    configs = {
        "model_a_fpn": {
            "name": "model_a_fpn",
            "num_classes": 1,
            "base_channels": 32,
            "neck_type": "fpn",
            "use_cbam": False,
            "use_ema": False,
            "strides": [8, 16, 32],
        },
        "model_b_pan": {
            "name": "model_b_pan",
            "num_classes": 1,
            "base_channels": 32,
            "neck_type": "fpnpan",
            "use_cbam": False,
            "use_ema": False,
            "strides": [8, 16, 32],
        },
        "model_c_attn": {
            "name": "model_c_attn",
            "num_classes": 1,
            "base_channels": 32,
            "neck_type": "fpnpan",
            "use_cbam": True,
            "use_ema": False,
            "strides": [8, 16, 32],
        },
        "model_d_p2_ema": {
            "name": "model_d_p2_ema",
            "num_classes": 1,
            "base_channels": 32,
            "neck_type": "fpnpan4",
            "use_cbam": True,
            "use_ema": True,
            "strides": [4, 8, 16, 32],
        }
    }
    
    m_info = configs[model_variant]
    
    cfg = {
        "model": {
            "name": m_info["name"],
            "num_classes": m_info["num_classes"],
            "base_channels": m_info["base_channels"],
            "neck_type": m_info["neck_type"],
            "use_cbam": m_info["use_cbam"],
            "use_ema": m_info["use_ema"],
        },
        "data": {
            "train_manifest": "/kaggle/working/splits/train.txt",
            "val_manifest": "/kaggle/working/splits/val.txt",
            "input_size": 416,
            "strides": m_info["strides"],
            "num_workers": 2,            # Safe 2 workers for fast target assignment without 30GB RAM OOM
            "persistent_workers": False, # Free RAM between epochs
            "prefetch_factor": 2,
        },
        "augmentation": {
            "random_hflip": True,
            "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
            "multi_scale": False
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": 0.0001,
            "warmup_epochs": 3,
            "grad_clip_norm": 10.0,
            "amp": True,
            "use_ema": m_info["use_ema"],
            "save_dir": f"/kaggle/working/runs/{model_variant}",
            "save_interval": 10
        },
        "loss": {
            "lambda_cls": 1.0,
            "lambda_reg": 2.0,
            "lambda_obj": 1.0,
            "alpha_hybrid": 0.5,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "nwd_constant": 12.8
        },
        "scheduler": {
            "type": "cosine",
            "eta_min": 1e-6
        },
        "wandb": {
            "enabled": False,
            "project": "vanilla-drone-detection",
            "run_name": model_variant
        }
    }
    return cfg

def train_model(model_variant, epochs=50, batch_size=16, lr=0.001):
    # Executes training using DDP by default on multi-GPU, with automatic fallback to DataParallel/Single-GPU
    import os, sys, subprocess, torch, gc, yaml
    from pathlib import Path
    
    cfg = get_config(model_variant, epochs=epochs, batch_size=batch_size, lr=lr)
    
    # Save config to file for torchrun
    cfg_dir = Path("/kaggle/working/configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{model_variant}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
        
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_ddp = num_gpus > 1
    
    if use_ddp:
        print("=" * 65)
        print(f"  [Multi-GPU] Launching DistributedDataParallel (DDP) on {num_gpus} GPUs")
        print("  Runner       : torchrun (NCCL Backend + DistributedSampler)")
        print(f"  Model Variant: {model_variant} | Epochs: {epochs} | Batch/GPU: {batch_size}")
        print("=" * 65)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            "/kaggle/working/train.py",
            "--config", str(cfg_path)
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[!] torchrun DDP exited with code {res.returncode}. Falling back to in-process DataParallel execution...")
            trainer = Trainer(cfg)
            trainer.run()
            del trainer
    else:
        print("=" * 65)
        print(f"  [Single-GPU/CPU] Launching in-process Trainer for {model_variant}")
        print("=" * 65)
        trainer = Trainer(cfg)
        trainer.run()
        del trainer
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def zip_model_run(run_dir_name, zip_name):
    # Helper to zip a model's run folder and generate interactive 1-click download link
    import zipfile
    from IPython.display import FileLink, display, HTML
    
    src_dir = Path(f"/kaggle/working/runs/{run_dir_name}")
    out_zip = Path(f"/kaggle/working/{zip_name}")
    
    if not src_dir.exists():
        print(f"[!] Directory {src_dir} does not exist. Skipping zip.")
        return
        
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_dir.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(Path("/kaggle/working")))
                
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[✓] {zip_name} packaged ({size_mb:.2f} MB). Click below to download:")
    display(FileLink(out_zip.name))
    display(HTML(f'<a href="{out_zip.name}" download style="display:inline-block; padding:6px 12px; background-color:#20beff; color:white; font-weight:bold; border-radius:4px; text-decoration:none; margin:6px 0;">⬇️ Download {zip_name}</a>'))
""")

    # Step 5.5: Checkpoint & Run Importer / Restorer
    md("""## 5.5 Checkpoint & Run Importer (Restore Previous Runs / Sessions)
Use this cell if you already trained models in previous sessions or downloaded `model_*_artifacts.zip` files.
- Automatically searches `/kaggle/input/` and `/kaggle/working/` for existing `runs/`, `.zip` archives, or `.pth` checkpoints.
- Unpacks them into `/kaggle/working/runs/` so you can jump straight to evaluation, curves, or benchmark tables without re-training!""")
    code(r"""import os, zipfile, glob, shutil
from pathlib import Path

runs_dir = Path("/kaggle/working/runs")
runs_dir.mkdir(parents=True, exist_ok=True)

# 1. Search for any uploaded model artifact zips in /kaggle/input or /kaggle/working
zip_candidates = glob.glob("/kaggle/input/**/model_*_artifacts.zip", recursive=True) + \
                 glob.glob("/kaggle/input/**/*artifacts*.zip", recursive=True) + \
                 glob.glob("/kaggle/working/model_*_artifacts.zip")

print(f"[*] Found {len(zip_candidates)} candidate artifact archives.")
for z_path in set(zip_candidates):
    try:
        print(f"[*] Extracting archive: {z_path} ...")
        with zipfile.ZipFile(z_path, 'r') as zf:
            zf.extractall("/kaggle/working")
    except Exception as e:
        print(f"[!] Error unpacking {z_path}: {e}")

# 2. Search for direct runs folders in /kaggle/input
for run_variant in ["model_a_fpn", "model_b_pan", "model_c_attn", "model_d_p2_ema", "baselines"]:
    src_matches = glob.glob(f"/kaggle/input/**/{run_variant}", recursive=True)
    for src_m in src_matches:
        dst = runs_dir / run_variant
        if not dst.exists():
            print(f"[*] Copying {src_m} -> {dst} ...")
            shutil.copytree(src_m, dst, dirs_exist_ok=True)

# 3. Print inventory of available checkpoints on disk
print("\n" + "=" * 65)
print("             AVAILABLE MODEL RUNS IN WORKSPACE")
print("=" * 65)
found_any = False
for var_dir in sorted(runs_dir.glob("*")):
    if var_dir.is_dir():
        best_p = var_dir / "best.pth"
        hist_p = var_dir / "history.json"
        sum_p = var_dir / "train_summary.json"
        
        status_items = []
        if best_p.exists(): status_items.append("best.pth (Weight Checkpoint)")
        if hist_p.exists(): status_items.append("history.json (Epoch Curves)")
        if sum_p.exists(): status_items.append("train_summary.json (Time/Metrics)")
        
        print(f"  ✓ {var_dir.name:<22}: {', '.join(status_items) if status_items else 'Folder present'}")
        found_any = True

if not found_any:
    print("  (No previous checkpoints found. Proceed to train models below)")
print("=" * 65)
""")

    # Step 6: Train Model-A (FPN Baseline)
    md("""## 6. Train Model-A (FPN Baseline)
Baseline single-stage anchor-free detector with top-down Feature Pyramid Network.
- Uses **DistributedDataParallel (DDP)** by default on dual GPUs, with automatic fallback.
- Set `RUN_MODEL_A = True` to train, or `False` to skip if using imported weights.""")
    code(r"""RUN_MODEL_A = True  # Set to False to skip training

if RUN_MODEL_A:
    train_model("model_a_fpn", epochs=50, batch_size=16, lr=0.001)
    zip_model_run("model_a_fpn", "model_a_artifacts.zip")
else:
    print("[*] Model-A training skipped (RUN_MODEL_A = False). Using existing run if available.")
""")

    # Step 7: Train Model-B (FPN + PAN)
    md("""## 7. Train Model-B (FPN + PAN Neck)
Introduces a bottom-up Path Aggregation Network (PAN) for bidirectional multi-scale feature propagation.
- Uses **DistributedDataParallel (DDP)** by default on dual GPUs, with automatic fallback.
- Set `RUN_MODEL_B = True` to train, or `False` to skip if using imported weights.""")
    code(r"""RUN_MODEL_B = True  # Set to False to skip training

if RUN_MODEL_B:
    train_model("model_b_pan", epochs=50, batch_size=16, lr=0.001)
    zip_model_run("model_b_pan", "model_b_artifacts.zip")
else:
    print("[*] Model-B training skipped (RUN_MODEL_B = False). Using existing run if available.")
""")

    # Step 8: Train Model-C (FPN + PAN + CBAM Attention)
    md("""## 8. Train Model-C (FPN + PAN + CBAM Attention)
Integrates Convolutional Block Attention Modules (CBAM) into CSP backbone stages for channel and spatial focus.
- Uses **DistributedDataParallel (DDP)** by default on dual GPUs, with automatic fallback.
- Set `RUN_MODEL_C = True` to train, or `False` to skip if using imported weights.""")
    code(r"""RUN_MODEL_C = True  # Set to False to skip training

if RUN_MODEL_C:
    train_model("model_c_attn", epochs=50, batch_size=16, lr=0.001)
    zip_model_run("model_c_attn", "model_c_artifacts.zip")
else:
    print("[*] Model-C training skipped (RUN_MODEL_C = False). Using existing run if available.")
""")

    # Step 9: Train Model-D (P2-P5 4-Level High-Res + CBAM + Model EMA)
    md("""## 9. Train Model-D (P2-P5 4-Level High-Res Neck + CBAM + Model EMA)
Advanced architecture featuring a Stride-4 (P2) high-resolution detection level for sub-20px drones, CBAM attention, and Model EMA weight smoothing.
- Uses **DistributedDataParallel (DDP)** by default on dual GPUs, with automatic fallback.
- Set `RUN_MODEL_D = True` to train, or `False` to skip if using imported weights.""")
    code(r"""RUN_MODEL_D = True  # Set to False to skip training

if RUN_MODEL_D:
    train_model("model_d_p2_ema", epochs=50, batch_size=16, lr=0.001)
    zip_model_run("model_d_p2_ema", "model_d_artifacts.zip")
else:
    print("[*] Model-D training skipped (RUN_MODEL_D = False). Using existing run if available.")
""")

    # Step 10: Plot Training & Ablation Curves
    md("""## 10. Plot Convergence & Loss Curves
Automatically loads training history from in-memory variables or saved `history.json` files on disk across all trained/imported variants.""")
    code(r"""import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_history(var_name, run_dir_name):
    if var_name in globals() and globals()[var_name]:
        return globals()[var_name]
    disk_path = Path(f"/kaggle/working/runs/{run_dir_name}/history.json")
    if disk_path.exists():
        with open(disk_path) as f:
            print(f"[*] Loaded history from disk: {disk_path}")
            return json.load(f)
    return None

hist_a = load_history("history_a", "model_a_fpn")
hist_b = load_history("history_b", "model_b_pan")
hist_c = load_history("history_c", "model_c_attn")
hist_d = load_history("history_d", "model_d_p2_ema")

models_found = [
    ("Model-A (FPN)", hist_a),
    ("Model-B (FPN+PAN)", hist_b),
    ("Model-C (FPN+PAN+CBAM)", hist_c),
    ("Model-D (P2-P5+CBAM+EMA)", hist_d)
]
active_models = [(name, h) for name, h in models_found if h is not None]

if active_models:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    for i, (m_name, h) in enumerate(active_models):
        ep = [x["epoch"] for x in h]
        ls = [x["total_loss"] for x in h]
        m50 = [x["mAP50"] for x in h]
        c = colors[i % len(colors)]
        axes[0].plot(ep, ls, label=m_name, color=c, linewidth=2)
        axes[1].plot(ep, m50, label=m_name, color=c, linewidth=2)

    axes[0].set_title('Ablation: Training Loss Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Total Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    axes[1].set_title('Ablation: Validation mAP@50 Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('mAP@50', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('/kaggle/working/ablation_curves.png', dpi=300)
    plt.show()
    print("[✓] Saved ablation curves to: /kaggle/working/ablation_curves.png")
else:
    print("[!] No training histories found on disk or in memory.")
""")

    # Step 11: Quantitative Evaluation & Best Checkpoint Metrics
    md("""## 11. Quantitative Evaluation with Checkpoint Manager
Evaluates all available model checkpoints on disk, measures latency & FPS, and dynamically determines the top-performing model.""")
    code(r"""import os, json
import torch
from src.utils.checkpoint import CheckpointManager
from src.data.dataset import DroneYOLODataset
from torch.utils.data import DataLoader
from src.engine.target_assigner import TargetAssigner
from src.models.detector import VanillaDroneDetector
from src.engine.evaluator import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluated_metrics = {}

variants = [
    ("Model-A (FPN)", "model_a_fpn", "fpn", False, [8, 16, 32]),
    ("Model-B (FPN+PAN)", "model_b_pan", "fpnpan", False, [8, 16, 32]),
    ("Model-C (FPN+PAN+CBAM)", "model_c_attn", "fpnpan", True, [8, 16, 32]),
    ("Model-D (P2-P5+CBAM+EMA)", "model_d_p2_ema", "fpnpan4", True, [4, 8, 16, 32]),
]

val_ds = DroneYOLODataset("/kaggle/working/splits/val.txt", input_size=416)

for label, var_name, neck, use_cbam, strides in variants:
    ckpt_path = f"/kaggle/working/runs/{var_name}/best.pth"
    if os.path.exists(ckpt_path):
        assigner = TargetAssigner(input_size=416, strides=strides, num_classes=1)
        val_loader = DataLoader(
            val_ds, batch_size=16, shuffle=False,
            num_workers=2, collate_fn=assigner.collate_and_assign
        )
        
        cfg_m = {"num_classes": 1, "base_channels": 32, "neck_type": neck, "use_cbam": use_cbam}
        model = VanillaDroneDetector(cfg_m).to(device)
        state = CheckpointManager.load(ckpt_path, model, device=device)
        evaluator = Evaluator(model, val_loader, num_classes=1, device=device)
        metrics = evaluator.evaluate()
        
        # Load training duration if available
        train_sum_p = Path(f"/kaggle/working/runs/{var_name}/train_summary.json")
        train_time_str = "N/A"
        if train_sum_p.exists():
            with open(train_sum_p) as f:
                train_time_str = json.load(f).get("total_train_time_formatted", "N/A")
        metrics["train_time"] = train_time_str
        
        evaluated_metrics[label] = metrics
        
        print(f"\n=============================================")
        print(f"  {label} (Epoch {state['epoch']}) VALIDATION METRICS")
        print(f"=============================================")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<15}: {v:.4f}")
            else:
                print(f"  {k:<15}: {v}")
        print(f"=============================================")

# Dynamically select best custom model based on mAP50
best_custom_label = None
best_custom_map = -1.0
best_custom_model = None

for label, m_data in evaluated_metrics.items():
    if m_data.get("mAP50", 0.0) > best_custom_map:
        best_custom_map = m_data["mAP50"]
        best_custom_label = label

if best_custom_label:
    print(f"\n[★] Best Custom Model: {best_custom_label} with mAP@50 = {best_custom_map*100:.2f}%")
    for label, var_name, neck, use_cbam, strides in variants:
        if label == best_custom_label:
            best_custom_model = VanillaDroneDetector({"num_classes": 1, "base_channels": 32, "neck_type": neck, "use_cbam": use_cbam}).to(device)
            CheckpointManager.load(f"/kaggle/working/runs/{var_name}/best.pth", best_custom_model, device=device)
            best_custom_model.eval()
            break
""")

    # Step 12: Visual Inference on Validation Samples
    md("## 12. Qualitative Detection Visualizations (Top Custom Model)")
    code(r"""import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.utils.box_ops import decode_predictions, batched_nms

if best_custom_model is not None:
    best_custom_model.eval()

    # Select 3 sample images from val split
    with open("/kaggle/working/splits/val.txt") as f:
        sample_images = [line.strip() for line in f if line.strip()][:3]

    fig, axes = plt.subplots(1, len(sample_images), figsize=(18, 6))

    for ax, img_path in zip(axes, sample_images):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img_resized = img.resize((416, 416), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            outs = best_custom_model(img_tensor)
        
        dets = decode_predictions(outs, conf_threshold=0.25)[0]
        dets = batched_nms(dets.cpu(), iou_threshold=0.45)
        
        draw = ImageDraw.Draw(img)
        scale_x = orig_w / 416.0
        scale_y = orig_h / 416.0
        
        for d in dets:
            x1, y1, x2, y2, conf, cls_id = d.tolist()
            draw.rectangle([x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y], outline="red", width=3)
            draw.text((x1*scale_x, max(0, y1*scale_y - 15)), f"drone {conf:.2f}", fill="red")
            
        ax.imshow(img)
        ax.set_title(Path(img_path).name[:30] + "...")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig('/kaggle/working/detection_samples.png', dpi=300)
    plt.show()
    print("[✓] Qualitative detections saved to: /kaggle/working/detection_samples.png")
else:
    print("[!] No custom model loaded for visualization.")
""")

    # Step 13: Pretrained Baselines (YOLOv8 & RT-DETR Vision Transformer)
    md("""## 13. Pretrained Baselines for Comparison Table (YOLOv8 & RT-DETR)
*Used solely to generate benchmark reference data for Section IV of the paper.*
- Each baseline model runs in its own independent cell with its own toggle flag and 1-click artifact zip download.""")

    # Step 13.0: Setup Baseline Config & Zipper
    md("### 13.0 Setup Baseline Dataset Config & Zipper Helper")
    code(r"""import yaml, json, time
from pathlib import Path
from ultralytics import YOLO, RTDETR

yolo_data = {
    "train": "/kaggle/working/splits/train.txt",
    "val": "/kaggle/working/splits/val.txt",
    "nc": 1,
    "names": {0: "drone"}
}

with open("/kaggle/working/drone_yolo.yaml", "w") as f:
    yaml.dump(yolo_data, f)

if "baseline_benchmarks" not in globals():
    baseline_benchmarks = {}

def zip_baseline_run(run_dir_name, zip_name):
    # Helper to zip a baseline run folder and generate interactive 1-click download link
    import zipfile
    from IPython.display import FileLink, display, HTML
    
    src_dir = Path(f"/kaggle/working/runs/baselines/{run_dir_name}")
    out_zip = Path(f"/kaggle/working/{zip_name}")
    
    if not src_dir.exists():
        print(f"[!] Directory {src_dir} does not exist. Skipping zip.")
        return
        
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_dir.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                zf.write(f, f.relative_to(Path("/kaggle/working")))
                
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[✓] {zip_name} packaged ({size_mb:.2f} MB). Click below to download:")
    display(FileLink(out_zip.name))
    display(HTML(f'<a href="{out_zip.name}" download style="display:inline-block; padding:6px 12px; background-color:#20beff; color:white; font-weight:bold; border-radius:4px; text-decoration:none; margin:6px 0;">⬇️ Download {zip_name}</a>'))
""")

    # Step 13.1: Train YOLOv8-nano
    md("### 13.1 Train YOLOv8-nano Baseline (~3.2M Params - Edge CNN)")
    code(r"""RUN_YOLO_NANO = True  # Set to False to skip

if RUN_YOLO_NANO:
    print("=" * 60)
    print("  TRAINING YOLOv8-nano BASELINE (~3.2M Params - Edge CNN)")
    print("=" * 60)
    t0 = time.perf_counter()
    yolo_n = YOLO("yolov8n.pt")
    yolo_n.train(
        data="/kaggle/working/drone_yolo.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        project="/kaggle/working/runs/baselines",
        name="yolov8n_baseline",
        device=0,
        workers=2,
        cache="ram",
        plots=False,
        exist_ok=True
    )
    t_train = time.perf_counter() - t0
    mins, secs = divmod(int(t_train), 60)
    
    val_n = yolo_n.val(plots=True)
    lat_n = val_n.speed.get("inference", 0.0)
    m_data = {
        "mAP50": float(val_n.box.map50),
        "mAP50_95": float(val_n.box.map),
        "precision": float(val_n.box.mp),
        "recall": float(val_n.box.mr),
        "latency_ms": lat_n,
        "fps": (1000.0 / lat_n) if lat_n > 0 else 0.0,
        "train_time": f"{mins}m {secs}s"
    }
    baseline_benchmarks["YOLOv8-nano Baseline"] = m_data
    
    # Save metrics to disk
    m_path = Path("/kaggle/working/runs/baselines/yolov8n_baseline/metrics.json")
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w") as f:
        json.dump(m_data, f, indent=2)
        
    zip_baseline_run("yolov8n_baseline", "yolov8n_artifacts.zip")
else:
    print("[*] YOLOv8-nano training skipped. Using existing run if available.")
""")

    # Step 13.2: Train YOLOv8-small
    md("### 13.2 Train YOLOv8-small Baseline (~11.2M Params - Capacity Match)")
    code(r"""RUN_YOLO_SMALL = True  # Set to False to skip

if RUN_YOLO_SMALL:
    print("\n" + "=" * 60)
    print("  TRAINING YOLOv8-small BASELINE (~11.2M Params - Capacity Match)")
    print("=" * 60)
    t0 = time.perf_counter()
    yolo_s = YOLO("yolov8s.pt")
    yolo_s.train(
        data="/kaggle/working/drone_yolo.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        project="/kaggle/working/runs/baselines",
        name="yolov8s_baseline",
        device=0,
        workers=2,
        cache="ram",
        plots=False,
        exist_ok=True
    )
    t_train = time.perf_counter() - t0
    mins, secs = divmod(int(t_train), 60)
    
    val_s = yolo_s.val(plots=True)
    lat_s = val_s.speed.get("inference", 0.0)
    m_data = {
        "mAP50": float(val_s.box.map50),
        "mAP50_95": float(val_s.box.map),
        "precision": float(val_s.box.mp),
        "recall": float(val_s.box.mr),
        "latency_ms": lat_s,
        "fps": (1000.0 / lat_s) if lat_s > 0 else 0.0,
        "train_time": f"{mins}m {secs}s"
    }
    baseline_benchmarks["YOLOv8-small Baseline"] = m_data
    
    # Save metrics to disk
    m_path = Path("/kaggle/working/runs/baselines/yolov8s_baseline/metrics.json")
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w") as f:
        json.dump(m_data, f, indent=2)
        
    zip_baseline_run("yolov8s_baseline", "yolov8s_artifacts.zip")
else:
    print("[*] YOLOv8-small training skipped. Using existing run if available.")
""")

    # Step 13.3: Train RT-DETR-L
    md("### 13.3 Train RT-DETR-L Baseline (~32.0M Params - Vision Transformer)")
    code(r"""RUN_RTDETR = True  # Set to False to skip

if RUN_RTDETR:
    print("\n" + "=" * 60)
    print("  TRAINING RT-DETR-L BASELINE (~32.0M Params - Real-Time Transformer)")
    print("=" * 60)
    t0 = time.perf_counter()
    rtdetr_l = RTDETR("rtdetr-l.pt")
    rtdetr_l.train(
        data="/kaggle/working/drone_yolo.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        project="/kaggle/working/runs/baselines",
        name="rtdetr_l_baseline",
        device=0,
        workers=2,
        cache="ram",
        plots=False,
        exist_ok=True
    )
    t_train = time.perf_counter() - t0
    mins, secs = divmod(int(t_train), 60)
    
    val_detr = rtdetr_l.val(plots=True)
    lat_detr = val_detr.speed.get("inference", 0.0)
    m_data = {
        "mAP50": float(val_detr.box.map50),
        "mAP50_95": float(val_detr.box.map),
        "precision": float(val_detr.box.mp),
        "recall": float(val_detr.box.mr),
        "latency_ms": lat_detr,
        "fps": (1000.0 / lat_detr) if lat_detr > 0 else 0.0,
        "train_time": f"{mins}m {secs}s"
    }
    baseline_benchmarks["RT-DETR-L Baseline"] = m_data
    
    # Save metrics to disk
    m_path = Path("/kaggle/working/runs/baselines/rtdetr_l_baseline/metrics.json")
    m_path.parent.mkdir(parents=True, exist_ok=True)
    with open(m_path, "w") as f:
        json.dump(m_data, f, indent=2)
        
    zip_baseline_run("rtdetr_l_baseline", "rtdetr_l_artifacts.zip")
else:
    print("[*] RT-DETR-L training skipped. Using existing run if available.")
""")

    # Step 14: Comprehensive Benchmark Comparison Table & Charts
    md("""## 14. Master Model Comparison & Benchmark Summary
Generates a comparative table and performance bar charts across all trained or imported models.""")
    code(r"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model_metadata = {
    "Model-A (FPN)": {
        "Backbone": "CSPDarknet",
        "Neck": "FPN",
        "Attention": "None",
        "Params (M)": 13.25,
        "Weights": "Scratch (Vanilla)",
        "Type": "CNN",
    },
    "Model-B (FPN+PAN)": {
        "Backbone": "CSPDarknet",
        "Neck": "FPN + PAN",
        "Attention": "None",
        "Params (M)": 14.57,
        "Weights": "Scratch (Vanilla)",
        "Type": "CNN",
    },
    "Model-C (FPN+PAN+CBAM)": {
        "Backbone": "CSPDarknet",
        "Neck": "FPN + PAN",
        "Attention": "CBAM (Spatial+Ch)",
        "Params (M)": 14.62,
        "Weights": "Scratch (Vanilla)",
        "Type": "CNN",
    },
    "Model-D (P2-P5+CBAM+EMA)": {
        "Backbone": "CSPDarknet",
        "Neck": "FPN + PAN (4-Level)",
        "Attention": "CBAM + Model EMA",
        "Params (M)": 15.34,
        "Weights": "Scratch (Vanilla)",
        "Type": "CNN",
    },
    "YOLOv8-nano Baseline": {
        "Backbone": "Modified CSPNet",
        "Neck": "PAN",
        "Attention": "None",
        "Params (M)": 3.20,
        "Weights": "Pretrained (COCO)",
        "Type": "CNN",
    },
    "YOLOv8-small Baseline": {
        "Backbone": "Modified CSPNet",
        "Neck": "PAN",
        "Attention": "None",
        "Params (M)": 11.20,
        "Weights": "Pretrained (COCO)",
        "Type": "CNN",
    },
    "RT-DETR-L Baseline": {
        "Backbone": "HGNetv2",
        "Neck": "Hybrid Encoder",
        "Attention": "Transformer Cross-Attn",
        "Params (M)": 32.00,
        "Weights": "Pretrained (COCO)",
        "Type": "Vision Transformer",
    }
}

rows = []

baseline_dir_map = {
    "YOLOv8-nano Baseline": "yolov8n_baseline",
    "YOLOv8-small Baseline": "yolov8s_baseline",
    "RT-DETR-L Baseline": "rtdetr_l_baseline",
}

# Collect metrics from evaluated custom models and external baselines
for label, m_info in model_metadata.items():
    metrics = None
    if label in globals().get("evaluated_metrics", {}):
        metrics = evaluated_metrics[label]
    elif "baseline_benchmarks" in globals() and label in baseline_benchmarks:
        metrics = baseline_benchmarks[label]
    elif label in baseline_dir_map:
        m_file = Path(f"/kaggle/working/runs/baselines/{baseline_dir_map[label]}/metrics.json")
        if m_file.exists():
            with open(m_file) as f:
                metrics = json.load(f)
    
    if metrics is not None:
        lat = metrics.get("latency_ms", 0.0)
        fps = metrics.get("fps", 0.0)
        rows.append({
            "Model": label,
            "Paradigm": m_info["Type"],
            "Backbone": m_info["Backbone"],
            "Neck": m_info["Neck"],
            "Attention": m_info["Attention"],
            "Params (M)": f"{m_info['Params (M)']:.2f}M",
            "mAP@50 (%)": f"{metrics.get('mAP50', 0.0) * 100:.2f}%",
            "mAP@50:95 (%)": f"{metrics.get('mAP50_95', 0.0) * 100:.2f}%",
            "Precision (%)": f"{metrics.get('precision', 0.0) * 100:.2f}%",
            "Recall (%)": f"{metrics.get('recall', 0.0) * 100:.2f}%",
            "Train Time": metrics.get("train_time", "N/A"),
            "Latency (ms)": f"{lat:.2f} ms" if lat > 0 else "N/A",
            "FPS": f"{fps:.1f}" if fps > 0 else "N/A",
            "Training Type": m_info["Weights"],
        })

df = pd.DataFrame(rows)

print("=" * 110)
print("                           MASTER MODEL BENCHMARK & COMPARISON TABLE")
print("=" * 110)
if not df.empty:
    print(df.to_string(index=False))
else:
    print("[!] No models evaluated yet. Run training or import previous runs first.")
print("=" * 110)

# Export to CSV and Markdown
csv_path = "/kaggle/working/model_comparison_table.csv"
md_path = "/kaggle/working/model_comparison_table.md"
if not df.empty:
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))
    print(f"[✓] Comparison table exported to:\n    - {csv_path}\n    - {md_path}")

# Generate Publication-Quality 2-Panel Figure: Benchmark Bar Chart + Pareto Frontier
if len(rows) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6.5))
    
    # ── Panel 1: Multi-Architecture Accuracy Comparison ────────────────────────
    models = [
        r["Model"].replace(" (P2-P5+CBAM+EMA)", "\n(Model-D)")
                  .replace(" (FPN+PAN+CBAM)", "\n(Model-C)")
                  .replace(" (FPN+PAN)", "\n(Model-B)")
                  .replace(" (FPN)", "\n(Model-A)")
                  .replace(" Baseline", "")
        for r in rows
    ]
    map50_vals = [float(r["mAP@50 (%)"].replace("%", "")) for r in rows]
    map50_95_vals = [float(r["mAP@50:95 (%)"].replace("%", "")) for r in rows]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, map50_vals, width, label='mAP@50 (%)', color='#2b5c8f', edgecolor='black', alpha=0.9)
    rects2 = ax1.bar(x + width/2, map50_95_vals, width, label='mAP@50:95 (%)', color='#e05a47', edgecolor='black', alpha=0.9)
    
    ax1.set_ylabel('mAP Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Detection Accuracy Across Architectures', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9.0, fontweight='bold')
    ax1.legend(fontsize=10.5, loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.set_ylim(0, 105)
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # ── Panel 2: Speed-Accuracy Pareto Frontier (FPS vs mAP@50:95) ─────────────
    fps_vals = []
    map_strict = []
    labels = []
    params = []
    colors_scatter = []
    
    color_map = {
        "Model-A": "#1f77b4",
        "Model-B": "#ff7f0e",
        "Model-C": "#2ca02c",
        "Model-D": "#9467bd",
        "YOLOv8-nano": "#d62728",
        "YOLOv8-small": "#8c564b",
        "RT-DETR-L": "#e377c2"
    }

    for r in rows:
        fps_str = r.get("FPS", "N/A")
        if fps_str != "N/A":
            fps_val = float(fps_str)
            m_strict = float(r["mAP@50:95 (%)"].replace("%", ""))
            p_val = float(r["Params (M)"].replace("M", ""))
            m_label = r["Model"].split(" (")[0].replace(" Baseline", "")
            
            fps_vals.append(fps_val)
            map_strict.append(m_strict)
            labels.append(m_label)
            params.append(p_val)
            colors_scatter.append(color_map.get(m_label, "#333333"))

    if fps_vals:
        bubble_sizes = [max(p * 25, 80) for p in params]
        scatter = ax2.scatter(fps_vals, map_strict, s=bubble_sizes, c=colors_scatter, alpha=0.75, edgecolors='black', linewidth=1.5)
        
        # Dedicated non-overlapping offsets for crystal-clear publication labels
        label_offsets = {
            "Model-A": (15, -2.5),
            "Model-B": (-120, -2.5),
            "Model-C": (15, 2.2),
            "Model-D": (18, -1.2),
            "YOLOv8-nano": (-125, 1.8),
            "YOLOv8-small": (-45, -3.2),
            "RT-DETR-L": (18, -0.5),
        }
        
        for i, txt in enumerate(labels):
            dx, dy = label_offsets.get(txt, (12, 1.5))
            ax2.annotate(
                f"{txt} ({params[i]:.1f}M)",
                (fps_vals[i], map_strict[i]),
                xytext=(fps_vals[i] + dx, map_strict[i] + dy),
                fontsize=8.5,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.9)
            )

        ax2.set_xlabel('Inference Speed (FPS) [Higher is Faster →]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Strict COCO mAP@50:95 (%) [Higher is Better ↑]', fontsize=12, fontweight='bold')
        ax2.set_title('(b) Efficiency vs. Precision Pareto Frontier', fontsize=13, fontweight='bold', pad=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_ylim(min(map_strict) - 6, max(map_strict) + 5)
        
        # Draw real-time threshold line at 30 FPS
        ax2.axvline(x=30, color='#888888', linestyle=':', linewidth=1.5)
        ax2.text(33, min(map_strict) - 4.5, "Real-Time Threshold (30 FPS)", fontsize=8.0, color='#666666', style='italic')

    plt.tight_layout()
    chart_path = "/kaggle/working/model_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.show()
    print(f"[✓] 2-Panel master benchmark figure saved to: {chart_path}")
""")

    # Step 15: Final Master Comparison Archive Zip
    md("""## 15. Master Comparison & Summary Archive Download
Creates a clean archive containing only comparative summary results: comparison CSV/Markdown tables, performance charts, ablation loss curves, qualitative detection plots, and split manifests.""")
    code(r"""import os
import zipfile
from pathlib import Path
from IPython.display import FileLink, display, HTML

working_dir = Path("/kaggle/working")
master_zip_path = working_dir / "master_comparison_artifacts.zip"

summary_artifacts = []

# 1. Comparison tables and documentation
for f in ["model_comparison_table.csv", "model_comparison_table.md", "drone_yolo.yaml"]:
    p = working_dir / f
    if p.exists():
        summary_artifacts.append(p)

# 2. Comparison figures & plots
for f in ["ablation_curves.png", "model_comparison_chart.png", "detection_samples.png"]:
    p = working_dir / f
    if p.exists():
        summary_artifacts.append(p)

# 3. Splits
splits_dir = working_dir / "splits"
if splits_dir.exists():
    for p in splits_dir.glob("*.txt"):
        summary_artifacts.append(p)

# 4. Per-model summary jsons
for p in working_dir.glob("runs/**/train_summary.json"):
    summary_artifacts.append(p)
for p in working_dir.glob("runs/**/history.json"):
    summary_artifacts.append(p)

summary_artifacts = sorted(list(set(summary_artifacts)))

print(f"[*] Packaging {len(summary_artifacts)} summary files into: {master_zip_path.name} ...")

with zipfile.ZipFile(master_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in summary_artifacts:
        try:
            rel_path = file_path.relative_to(working_dir)
        except ValueError:
            rel_path = file_path.name
        zf.write(file_path, arcname=str(rel_path))

zip_size_mb = master_zip_path.stat().st_size / (1024 * 1024)

print("\n" + "=" * 65)
print("        MASTER COMPARISON SUMMARY ARCHIVE READY")
print("=" * 65)
print(f"  Archive Name : {master_zip_path.name}")
print(f"  Archive Size : {zip_size_mb:.2f} MB")
print(f"  Files Bundled: {len(summary_artifacts)}")
print("=" * 65)

display(FileLink(master_zip_path.name))
display(HTML(f'<a href="{master_zip_path.name}" download style="display:inline-block; padding:8px 16px; background-color:#2ca02c; color:white; font-weight:bold; border-radius:4px; text-decoration:none; margin-top:10px;">⬇️ Download Master Comparison Artifacts (.zip)</a>'))
""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = root / "kaggle" / "drone_detector_kaggle.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"[OK] Notebook generated at: {out_path}")

if __name__ == "__main__":
    build_notebook()
