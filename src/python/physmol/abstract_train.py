"""CLI for PHYSMOL abstract cognition training."""

from __future__ import annotations

import argparse
import os
from typing import List

from .abstract_training import AbstractCognitionTrainer
from .progress import ProgressLogger
from .training_data import (
    LocalDatasetAdapter,
    ModelScopeDatasetAdapter,
    TrainingExample,
)


def build_builtin_examples() -> List[TrainingExample]:
    return [
        TrainingExample(
            modality="abstract",
            text="prove even + even is even",
            target="Let the numbers be 2a and 2b. Their sum is 2(a+b), so it is even.",
            metadata={"task": "math_proof"},
        ),
        TrainingExample(
            modality="abstract",
            text="A crime with evidence and intent: what punishment is justified?",
            target="Apply due process, verify evidence and intent, then use proportionate punishment.",
            metadata={"task": "legal_case"},
        ),
        TrainingExample(
            modality="abstract",
            text="Slavery is wrong because it denies freedom and equality.",
            target="Avoid enslave_or_dominate_person actions because they violate freedom and equality.",
            metadata={"task": "moral_value"},
        ),
    ]


def load_examples(args) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    local = LocalDatasetAdapter()

    if args.local_jsonl:
        examples.extend(local.load_jsonl(args.local_jsonl, modality="abstract", limit=args.limit))
    if args.local_text:
        examples.extend(local.load_text(args.local_text, modality="abstract", limit=args.limit))
    if args.modelscope_dataset:
        ms = ModelScopeDatasetAdapter()
        examples.extend(ms.load(
            args.modelscope_dataset,
            subset_name=args.subset_name or None,
            split=args.split,
            limit=args.limit,
        ))
    if args.use_builtin or not examples:
        examples.extend(build_builtin_examples())

    return examples[:args.limit] if args.limit else examples


def main():
    parser = argparse.ArgumentParser(description="Train PHYSMOL abstract cognition components")
    parser.add_argument("--modelscope-dataset", default="", help="ModelScope dataset id, e.g. clue")
    parser.add_argument("--subset-name", default="", help="ModelScope subset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--local-jsonl", default="", help="Local JSONL dataset path")
    parser.add_argument("--local-text", default="", help="Local plain text dataset path")
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples to load")
    parser.add_argument("--out-dir", default="./checkpoints/abstract_training")
    parser.add_argument("--use-builtin", action="store_true", help="Include built-in seed examples")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress = ProgressLogger(args.out_dir)
    examples = load_examples(args)

    trainer = AbstractCognitionTrainer()
    progress.log(0, len(examples), "abstract_training", message="Starting abstract training")
    metrics = trainer.train(examples, progress=progress)
    trainer.save(args.out_dir)
    progress.log(len(examples), len(examples), "complete", metrics=metrics,
                 message=f"Saved abstract cognition artifacts to {args.out_dir}")
    print(metrics)


if __name__ == "__main__":
    main()
