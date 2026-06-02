"""Dialogue Training Pipeline for PHYSMOL.

Trains the Broca language production module using dialogue data.

Supports:
  - ModelScope datasets (auto-download)
  - Local JSONL/JSON files
  - Built-in seed dialogues
  - Continuous learning from interaction

Usage:
    # Auto-download from ModelScope and train
    python -m physmol.dialogue_trainer --modelscope belle --epochs 3

    # Train from local file
    python -m physmol.dialogue_trainer --local data/train.jsonl

    # Use built-in seed data
    python -m physmol.dialogue_trainer --use-builtin
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Any

from .language.broca import BrocaModule
from .progress import ProgressLogger


# ---------------------------------------------------------------------------
# Built-in Seed Dialogues
# ---------------------------------------------------------------------------

def build_builtin_dialogues() -> List[Dict[str, str]]:
    """Built-in seed dialogues for basic language capability."""
    return [
        # Greetings
        {"input": "你好", "output": "你好！有什么我可以帮助你的吗？", "intent": "greeting"},
        {"input": "hello", "output": "Hello! How can I help you?", "intent": "greeting"},
        {"input": "你是谁", "output": "我是 PHYSMOL，一个基于物理认知的智能系统。", "intent": "identity"},

        # Physics
        {"input": "球为什么会下落", "output": "因为重力的作用，物体受到向下的力。", "intent": "explanation",
         "concepts": ["ball", "gravity", "fall"]},
        {"input": "什么是弹性", "output": "弹性是材料在变形后恢复原状的能力。", "intent": "explanation",
         "concepts": ["elasticity", "material", "deformation"]},
        {"input": "如果球更重会怎样", "output": "更重的球惯性更大，更难加速，但下落速度相同。", "intent": "counterfactual",
         "concepts": ["ball", "mass", "inertia", "gravity"]},

        # Code
        {"input": "什么是快速排序", "output": "快速排序是一种分治排序算法，选择基准元素分区后递归排序。", "intent": "explanation",
         "concepts": ["quicksort", "sort", "divide", "conquer"]},
        {"input": "解释二分查找", "output": "二分查找在有序数组中反复将搜索区间减半，时间复杂度为 O(log n)。", "intent": "explanation",
         "concepts": ["binary_search", "search", "logarithmic"]},
        {"input": "BFS和DFS有什么区别", "output": "BFS用队列逐层遍历，找最短路径；DFS用栈深入遍历，适合路径查找。", "intent": "comparison",
         "concepts": ["bfs", "dfs", "graph", "traverse"]},

        # General
        {"input": "你能做什么", "output": "我可以解释物理概念、生成代码、回答问题，并且在不断学习新知识。", "intent": "capability"},
        {"input": "谢谢", "output": "不客气！如果还有问题随时问我。", "intent": "closing"},
        {"input": "再见", "output": "再见！期待下次交流。", "intent": "closing"},
    ]


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

class DatasetLoader:
    """Load training data from various sources.

    Supported sources:
      - ModelScope: modelscope dataset ID
      - HuggingFace: huggingface dataset ID
      - Local files: JSONL, JSON, TXT
      - URLs: direct download links
    """

    @staticmethod
    def load_modelscope(dataset_id: str, subset: str = "",
                        split: str = "train", limit: int = 0) -> List[Dict]:
        """Load dataset from ModelScope."""
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError:
            print("modelscope not installed. Install with: pip install modelscope")
            return []

        print(f"Loading ModelScope dataset: {dataset_id}...")
        kwargs = {"dataset_name": dataset_id, "split": split}
        if subset:
            kwargs["subset_name"] = subset

        ds = MsDataset.load(**kwargs)

        dialogues = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            # Normalize different dataset formats
            dialogue = DatasetLoader._normalize_row(row)
            if dialogue:
                dialogues.append(dialogue)

        print(f"Loaded {len(dialogues)} dialogues from ModelScope")
        return dialogues

    @staticmethod
    def load_huggingface(dataset_id: str, subset: str = "",
                         split: str = "train", limit: int = 0) -> List[Dict]:
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            print("datasets not installed. Install with: pip install datasets")
            return []

        print(f"Loading HuggingFace dataset: {dataset_id}...")
        kwargs = {"path": dataset_id, "split": split}
        if subset:
            kwargs["name"] = subset

        try:
            ds = load_dataset(**kwargs, trust_remote_code=True)
        except TypeError:
            # Newer versions of datasets don't support trust_remote_code
            ds = load_dataset(**kwargs)

        dialogues = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            dialogue = DatasetLoader._normalize_row(row)
            if dialogue:
                dialogues.append(dialogue)

        print(f"Loaded {len(dialogues)} dialogues from HuggingFace")
        return dialogues

    @staticmethod
    def load_from_url(url: str, limit: int = 0) -> List[Dict]:
        """Load dataset from a URL (JSONL or JSON)."""
        import urllib.request
        import tempfile

        print(f"Downloading from: {url}...")

        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            tmp_path = tmp.name

        # Determine format from URL
        if url.endswith(".json") or url.endswith(".jsonl"):
            dialogues = DatasetLoader.load_jsonl(tmp_path, limit)
        else:
            dialogues = DatasetLoader.load_jsonl(tmp_path, limit)

        # Clean up
        import os
        os.remove(tmp_path)

        return dialogues

    @staticmethod
    def load_jsonl(path: str, limit: int = 0) -> List[Dict]:
        """Load from JSONL file."""
        dialogues = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    dialogue = DatasetLoader._normalize_row(row)
                    if dialogue:
                        dialogues.append(dialogue)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(dialogues)} dialogues from {path}")
        return dialogues

    @staticmethod
    def load_json(path: str, limit: int = 0) -> List[Dict]:
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            dialogues = []
            for i, row in enumerate(data):
                if limit and i >= limit:
                    break
                dialogue = DatasetLoader._normalize_row(row)
                if dialogue:
                    dialogues.append(dialogue)
            return dialogues

        return []

    @staticmethod
    def load_txt(path: str, limit: int = 0) -> List[Dict]:
        """Load from plain text file (one sentence per line)."""
        dialogues = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if line and len(line) > 3:
                    dialogues.append({
                        "input": "",
                        "output": line,
                        "intent": "general",
                    })
        return dialogues

    @staticmethod
    def _normalize_row(row: Any) -> Optional[Dict]:
        """Normalize different dataset formats to standard format."""
        if not isinstance(row, dict):
            return None

        # Format 1: instruction/input/output
        if "instruction" in row:
            return {
                "input": row.get("instruction", "") + " " + row.get("input", ""),
                "output": row.get("output", ""),
                "intent": row.get("intent", "general"),
            }

        # Format 2: question/answer
        if "question" in row:
            return {
                "input": row["question"],
                "output": row.get("answer", row.get("response", "")),
                "intent": row.get("intent", "general"),
            }

        # Format 3: messages list (ShareGPT style)
        if "messages" in row and isinstance(row["messages"], list):
            user_msgs = []
            assistant_msgs = []
            for msg in row["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    user_msgs.append(content)
                elif role == "assistant":
                    assistant_msgs.append(content)

            if user_msgs and assistant_msgs:
                return {
                    "input": " ".join(user_msgs),
                    "output": " ".join(assistant_msgs),
                    "intent": "conversation",
                }

        # Format 4: input/output
        if "input" in row and "output" in row:
            return {
                "input": row["input"],
                "output": row["output"],
                "intent": row.get("intent", "general"),
            }

        # Format 5: text only
        if "text" in row:
            return {
                "input": "",
                "output": row["text"],
                "intent": "general",
            }

        return None


# ---------------------------------------------------------------------------
# Dialogue Trainer
# ---------------------------------------------------------------------------

class DialogueTrainer:
    """Train the Broca language production module.

    Training pipeline:
      1. Load dialogue data
      2. Build vocabulary
      3. Learn grammar patterns
      4. Learn concept-word mappings
      5. Save trained model
    """

    def __init__(self, vsa_dim: int = 4096, output_dir: str = "./checkpoints/broca"):
        self.vsa_dim = vsa_dim
        self.output_dir = output_dir

        # Initialize Broca module
        self.broca = BrocaModule(vsa_dim)

        # Progress tracking
        self.progress = ProgressLogger(output_dir)

    def train(self, dialogues: List[Dict[str, str]], epochs: int = 1):
        """Train from dialogue data."""
        print("=" * 60)
        print("Broca Language Production Training")
        print("=" * 60)
        print(f"Dialogues: {len(dialogues)}")
        print(f"Epochs: {epochs}")
        print(f"Output: {self.output_dir}")
        print()

        os.makedirs(self.output_dir, exist_ok=True)

        start_time = time.time()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Train one epoch
            self.broca.train_from_dialogue(dialogues)

            # Show progress
            stats = self.broca.get_stats()
            print(f"  Vocabulary: {stats['vocabulary_size']} words")
            print(f"  Patterns: {stats['grammar']['total_patterns']}")
            print(f"  Concept mappings: {stats['grammar']['concept_mappings']}")

            # Log progress
            self.progress.log(epoch + 1, epochs, "broca_training",
                            message=f"Epoch {epoch + 1}")

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.1f}s")

        # Save model
        model_path = os.path.join(self.output_dir, "model")
        self.broca.save(model_path)
        print(f"Model saved to: {model_path}")

        return self.broca

    def train_from_source(self, source: str, **kwargs):
        """Train from a data source.

        Args:
            source: data source identifier
                - "builtin": built-in seed dialogues
                - "modelscope:<dataset_id>": ModelScope dataset
                - "huggingface:<dataset_id>": HuggingFace dataset
                - "url:<URL>": direct URL to JSONL/JSON file
                - "<path>": local file path
        """
        dialogues = []

        if source == "builtin":
            dialogues = build_builtin_dialogues()

        elif source.startswith("modelscope:"):
            dataset_id = source.split(":", 1)[1]
            dialogues = DatasetLoader.load_modelscope(
                dataset_id,
                subset=kwargs.get("subset", ""),
                split=kwargs.get("split", "train"),
                limit=kwargs.get("limit", 0),
            )

        elif source.startswith("huggingface:"):
            dataset_id = source.split(":", 1)[1]
            dialogues = DatasetLoader.load_huggingface(
                dataset_id,
                subset=kwargs.get("subset", ""),
                split=kwargs.get("split", "train"),
                limit=kwargs.get("limit", 0),
            )

        elif source.startswith("url:"):
            url = source.split(":", 1)[1]
            dialogues = DatasetLoader.load_from_url(url, kwargs.get("limit", 0))

        elif os.path.exists(source):
            if source.endswith(".jsonl"):
                dialogues = DatasetLoader.load_jsonl(source, kwargs.get("limit", 0))
            elif source.endswith(".json"):
                dialogues = DatasetLoader.load_json(source, kwargs.get("limit", 0))
            elif source.endswith(".txt"):
                dialogues = DatasetLoader.load_txt(source, kwargs.get("limit", 0))

        if not dialogues:
            print(f"No data loaded from: {source}")
            return None

        return self.train(dialogues, epochs=kwargs.get("epochs", 1))

    def test(self, test_cases: Optional[List[Dict]] = None):
        """Test the trained model."""
        if test_cases is None:
            test_cases = [
                {"concepts": ["ball", "gravity"], "intent": "explanation"},
                {"concepts": ["sort", "algorithm"], "intent": "explanation"},
                {"concepts": ["hello"], "intent": "greeting"},
                {"concepts": ["physics", "simulation"], "intent": "general"},
            ]

        print("\n" + "=" * 60)
        print("Testing Broca Module")
        print("=" * 60)

        for case in test_cases:
            concepts = case.get("concepts", [])
            intent = case.get("intent", "general")

            result = self.broca.produce(concepts, intent)
            print(f"  Input: {concepts} (intent: {intent})")
            print(f"  Output: {result}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PHYSMOL Broca language production module")

    # Data source
    parser.add_argument("--modelscope", default="",
                        help="ModelScope dataset ID (e.g., 'BelleGroup/train_1M_CN')")
    parser.add_argument("--huggingface", default="",
                        help="HuggingFace dataset ID (e.g., 'tatsu-lab/alpaca')")
    parser.add_argument("--url", default="",
                        help="Direct URL to JSONL/JSON file")
    parser.add_argument("--subset", default="", help="Dataset subset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--local", default="", help="Local data file path")
    parser.add_argument("--use-builtin", action="store_true",
                        help="Use built-in seed dialogues")

    # Training options
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to load")
    parser.add_argument("--vsa-dim", type=int, default=4096, help="VSA dimension")
    parser.add_argument("--out-dir", default="./checkpoints/broca",
                        help="Output directory")

    # Test
    parser.add_argument("--test", action="store_true", help="Run test after training")

    args = parser.parse_args()

    # Initialize trainer
    trainer = DialogueTrainer(vsa_dim=args.vsa_dim, output_dir=args.out_dir)

    # Determine data source
    if args.use_builtin:
        trainer.train(build_builtin_dialogues(), epochs=args.epochs)
    elif args.modelscope:
        trainer.train_from_source(
            f"modelscope:{args.modelscope}",
            subset=args.subset,
            split=args.split,
            limit=args.limit,
            epochs=args.epochs,
        )
    elif args.huggingface:
        trainer.train_from_source(
            f"huggingface:{args.huggingface}",
            subset=args.subset,
            split=args.split,
            limit=args.limit,
            epochs=args.epochs,
        )
    elif args.url:
        trainer.train_from_source(
            f"url:{args.url}",
            limit=args.limit,
            epochs=args.epochs,
        )
    elif args.local:
        trainer.train_from_source(
            args.local,
            limit=args.limit,
            epochs=args.epochs,
        )
    else:
        print("No data source specified. Using built-in seed dialogues.")
        trainer.train(build_builtin_dialogues(), epochs=args.epochs)

    # Test if requested
    if args.test:
        trainer.test()


if __name__ == "__main__":
    main()
