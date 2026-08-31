import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from src.models.detector import VanillaDroneDetector

def export_onnx(config_path, checkpoint_path, output_path, input_size=416):
    print(f"[+] Loading config: {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("[+] Initializing VanillaDroneDetector...")
    model = VanillaDroneDetector(cfg)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[+] Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state_dict)
    else:
        print("[!] No checkpoint provided. Exporting architecture with initialized weights.")
        
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['cls_preds', 'reg_preds', 'obj_preds']
    )
    print(f"[✓] Successfully exported ONNX model ({output_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export VanillaDroneDetector to ONNX')
    parser.add_argument('--config', type=str, required=True, help='Path to model YAML config')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint .pth')
    parser.add_argument('--output', type=str, default='runs/model_d_p2_ema.onnx', help='Output ONNX path')
    parser.add_argument('--input_size', type=int, default=416, help='Input image resolution')
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.output, args.input_size)
