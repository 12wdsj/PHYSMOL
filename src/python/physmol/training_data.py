"""Training data adapters for language, memory, and physical experience.

ModelScope support is optional.  If the `modelscope` package is installed, this
module can load community datasets through `MsDataset.load`.  Otherwise the same
normalization path works with local JSONL or text files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional


@dataclass
class TrainingExample:
    modality: str
    text: str
    target: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelScopeDatasetAdapter:
    """Optional adapter around modelscope.msdatasets.MsDataset."""

    def load(
        self,
        dataset_id: str,
        subset_name: Optional[str] = None,
        split: str = "train",
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[TrainingExample]:
        try:
            from modelscope.msdatasets import MsDataset
        except ImportError as exc:
            raise ImportError(
                "ModelScope dataset loading requires `pip install modelscope`. "
                "You can still use LocalDatasetAdapter for JSONL/text files."
            ) from exc

        ds = MsDataset.load(
            dataset_id,
            subset_name=subset_name,
            split=split,
            **kwargs,
        )
        examples: List[TrainingExample] = []
        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            examples.append(normalize_dataset_row(row, source="modelscope"))
        return examples


class LocalDatasetAdapter:
    """Load local JSONL or plain text files into TrainingExample objects."""

    def load_jsonl(
        self,
        path: str,
        modality: str = "language",
        limit: Optional[int] = None,
    ) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                examples.append(normalize_dataset_row(
                    json.loads(line), source=path, default_modality=modality))
        return examples

    def load_text(
        self,
        path: str,
        modality: str = "language",
        limit: Optional[int] = None,
    ) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                text = line.strip()
                if text:
                    examples.append(TrainingExample(
                        modality=modality,
                        text=text,
                        metadata={"source": path, "line": idx},
                    ))
        return examples


class PHYSMOLTrainingDataBuilder:
    """Route normalized examples into PHYSMOL subsystems."""

    def ingest_language_examples(self, cognitive_interface, examples: Iterable[TrainingExample]) -> dict:
        """Use text examples to grow vocabulary and long-term memory.

        This does not fine-tune a large decoder.  It trains PHYSMOL's symbolic
        interfaces by exposing the text encoder to terms, storing facts, and
        registering interaction examples for later supervised fine-tuning.
        """
        count = 0
        facts = 0
        for ex in examples:
            if ex.modality != "language":
                continue
            cognitive_interface.text_encoder.encode(ex.text)
            if ex.target:
                cognitive_interface.text_encoder.encode(ex.target)
            if hasattr(cognitive_interface, "long_term_memory"):
                cognitive_interface.long_term_memory.add_episode(
                    content=ex.text,
                    outcome=ex.target,
                    tags=["language", ex.metadata.get("task", "dialogue")],
                )
                if ex.metadata.get("task") == "fact":
                    cognitive_interface.long_term_memory.add_fact(
                        subject=ex.metadata.get("subject", "text"),
                        predicate=ex.metadata.get("predicate", "states"),
                        obj=ex.target or ex.text,
                        evidence=ex.text,
                        confidence=float(ex.metadata.get("confidence", 0.75)),
                        tags=["language", "fact"],
                    )
                    facts += 1
            count += 1
        return {"language_examples": count, "facts": facts}

    def ingest_physics_trace(self, cognitive_interface, trace: Iterable[dict]) -> dict:
        """Store simulator traces as long-term experience records."""
        count = 0
        for step in trace:
            if not hasattr(cognitive_interface, "long_term_memory"):
                break
            cognitive_interface.long_term_memory.add_experience(
                description=step.get("description", "physical interaction"),
                action=step.get("action"),
                observation=step.get("observation") or step.get("next_state"),
                reward=step.get("reward"),
                prediction_error=step.get("prediction_error"),
                tags=step.get("tags", ["physics"]),
            )
            count += 1
        return {"physics_steps": count}


def normalize_dataset_row(
    row: Any,
    source: str = "",
    default_modality: str = "language",
) -> TrainingExample:
    """Normalize common dataset row formats into TrainingExample."""
    if not isinstance(row, dict):
        return TrainingExample(
            modality=default_modality,
            text=str(row),
            metadata={"source": source},
        )

    if "messages" in row and isinstance(row["messages"], list):
        text_parts = []
        target = None
        for msg in row["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant":
                target = content
            else:
                text_parts.append(content)
        return TrainingExample(
            modality=row.get("modality", default_modality),
            text="\n".join(text_parts),
            target=target,
            metadata={**row, "source": source},
        )

    text = (
        row.get("instruction")
        or row.get("question")
        or row.get("query")
        or row.get("prompt")
        or row.get("text")
        or row.get("sentence1")
        or ""
    )
    if row.get("input"):
        text = f"{text}\n{row['input']}".strip()
    if row.get("sentence2"):
        text = f"{text}\n{row['sentence2']}".strip()

    target = (
        row.get("output")
        or row.get("answer")
        or row.get("response")
        or row.get("label")
    )
    if target is not None:
        target = str(target)

    modality = row.get("modality", default_modality)
    return TrainingExample(
        modality=modality,
        text=str(text),
        target=target,
        metadata={**row, "source": source},
    )
