# PHYSMOL on ModelScope Notebook

This guide assumes a ModelScope Notebook or Web-IDE environment.

## 1. Open The Cloud Notebook

1. Open `https://www.modelscope.cn/my/mynotebook`.
2. Start a CPU/GPU/ROCm notebook that fits the current phase.
3. Keep persistent project files under `/mnt/workspace`.

ModelScope Notebook is Jupyter-based and also provides a Web-IDE entry. The
official docs note that persistent PAI-DSW storage is mounted at
`/mnt/workspace`; paths outside that area can be cleared when the instance
stops.

## 2. Upload Or Clone PHYSMOL

If Git access is available:

```bash
cd /mnt/workspace
git clone <your-physmol-repo-url> PHYSMOL
cd PHYSMOL
```

If Git access is restricted, upload a zip through Notebook/Web-IDE and unpack:

```bash
cd /mnt/workspace
unzip PHYSMOL.zip -d PHYSMOL
cd PHYSMOL
```

## 3. Install Runtime Dependencies

For abstract cognition training:

```bash
python -m pip install -e ".[modelscope]"
```

For ROCm/PyTorch physics training, install the PyTorch build that is already
compatible with the Notebook image if present. In PyTorch, ROCm devices are
usually exposed through `cuda` APIs, so PHYSMOL accepts `--device rocm` but maps
it to the PyTorch CUDA device.

## 4. Train Abstract Cognition

Built-in seed data:

```bash
python -m physmol.abstract_train \
  --use-builtin \
  --out-dir /mnt/workspace/PHYSMOL/checkpoints/abstract_training
```

ModelScope dataset:

```bash
python -m physmol.abstract_train \
  --modelscope-dataset clue \
  --subset-name afqmc \
  --split train \
  --limit 1000 \
  --out-dir /mnt/workspace/PHYSMOL/checkpoints/abstract_training
```

Local JSONL:

```bash
python -m physmol.abstract_train \
  --local-jsonl /mnt/workspace/data/abstract_train.jsonl \
  --out-dir /mnt/workspace/PHYSMOL/checkpoints/abstract_training
```

Common JSONL row formats:

```json
{"instruction": "prove even + even is even", "output": "Let the numbers be 2a and 2b..."}
{"question": "A crime has evidence and intent. What punishment is justified?", "answer": "Use due process and proportionality."}
{"text": "Slavery is wrong because it denies freedom and equality.", "label": "avoid domination"}
```

## 5. Watch Training Progress

In a second Notebook terminal:

```bash
python -m physmol.progress_server \
  --host 0.0.0.0 \
  --port 7860 \
  --progress /mnt/workspace/PHYSMOL/checkpoints/abstract_training/progress.json
```

Then open the Notebook/Web-IDE port preview for port `7860` if your instance
provides one. If external port preview is unavailable, open a Notebook cell or
terminal and inspect:

```bash
watch -n 2 cat /mnt/workspace/PHYSMOL/checkpoints/abstract_training/progress.json
```

## 6. Train Physical Modules

Physics training uses simulator or analytical traces:

```bash
python -m physmol.unified_train \
  --phase 1 \
  --device rocm \
  --epochs 500 \
  --save-path /mnt/workspace/PHYSMOL/checkpoints
```

Language/VSA alignment:

```bash
python -m physmol.unified_train \
  --phase 3 \
  --device rocm \
  --phase3-epochs 100 \
  --save-path /mnt/workspace/PHYSMOL/checkpoints
```

End-to-end validation:

```bash
python -m physmol.unified_train \
  --phase 0 \
  --device rocm \
  --save-path /mnt/workspace/PHYSMOL/checkpoints
```
