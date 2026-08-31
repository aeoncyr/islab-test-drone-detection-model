"""
kaggle/generate_setup.py

Reads all local files under `src/` and generates `kaggle/setup_project.py`
using base64-encoded strings to avoid any multi-line quoting issues.
"""

import base64
from pathlib import Path

def generate():
    root = Path(__file__).resolve().parent.parent
    src_dir = root / "src"
    
    files_to_pack = []
    # Collect all python files in src
    for p in sorted(src_dir.rglob("*.py")):
        rel_path = p.relative_to(root).as_posix()
        raw_bytes = p.read_bytes()
        b64_str = base64.b64encode(raw_bytes).decode("ascii")
        files_to_pack.append((rel_path, b64_str))

    output = [
        '"""',
        'Self-contained bootstrap script generated from local src/',
        'Reconstructs the full src/ package inside /kaggle/working/',
        '"""',
        'import os, sys, base64',
        '',
        'ROOT = "/kaggle/working"',
        'if ROOT not in sys.path:',
        '    sys.path.insert(0, ROOT)',
        '',
        'FILES = {'
    ]

    for rel_path, b64_str in files_to_pack:
        output.append(f'    "{rel_path}": "{b64_str}",')

    output.extend([
        '}',
        '',
        'for rel_path, b64_str in FILES.items():',
        '    full_path = os.path.join(ROOT, rel_path)',
        '    os.makedirs(os.path.dirname(full_path), exist_ok=True)',
        '    with open(full_path, "wb") as f:',
        '        f.write(base64.b64decode(b64_str))',
        '    print(f"  [setup] wrote {rel_path}")',
        '',
        'print("[✓] All src/ modules successfully unpacked to /kaggle/working/src/")',
        ''
    ])

    setup_file = root / "kaggle" / "setup_project.py"
    setup_file.write_text("\n".join(output), encoding="utf-8")
    print(f"[OK] Generated {setup_file} from {len(files_to_pack)} python files.")

if __name__ == "__main__":
    generate()
