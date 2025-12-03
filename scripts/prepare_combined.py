"""Prepare combined dataset from tinyshakespeare + tinystories."""

import subprocess
import sys
from pathlib import Path

# Scripts and data directories
scripts_dir = Path(__file__).parent
data_dir = scripts_dir.parent / "data"
data_dir.mkdir(exist_ok=True)

# Prepare individual datasets if not present
if not (data_dir / "tinyshakespeare_train.txt").exists():
    print("Preparing TinyShakespeare...")
    subprocess.run([sys.executable, str(scripts_dir / "prepare_tinyshakespeare.py")])

if not (data_dir / "tinystories_train.txt").exists():
    print("Preparing TinyStories...")
    subprocess.run([sys.executable, str(scripts_dir / "prepare_tinystories.py")])

# Combine datasets
for split in ["train", "val"]:
    combined = []
    for dataset in ["tinyshakespeare", "tinystories"]:
        path = data_dir / f"{dataset}_{split}.txt"
        if path.exists():
            text = path.read_text()
            combined.append(text)
            print(f"Added {path.name}: {len(text):,} chars")

    out_path = data_dir / f"combined_{split}.txt"
    out_path.write_text("\n\n".join(combined))
    print(f"Wrote {out_path.name}: {len(out_path.read_text()):,} chars")

print("Done! Combined dataset written to data/")
