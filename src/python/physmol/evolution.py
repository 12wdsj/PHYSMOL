"""Evolution data recording and export for PHYSMOL.

This module turns normal interactions into training data that can later be used
to improve the language shell around PHYSMOL.  It deliberately keeps learning
artifacts separate from the cognitive modules:

* PHYSMOL keeps updating memories, concepts, and Broca patterns.
* The small LLM is improved offline from exported JSONL batches.

That separation prevents every casual interaction from immediately changing a
language model's weights while still making the whole system trainable.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def to_jsonable(value: Any) -> Any:
    """Convert numpy/dataclass-like values into JSON-friendly structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


@dataclass
class ToolTrace:
    """A single agent/tool interaction trace."""
    timestamp: float
    tool: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    user_input: str = ""
    final_response: str = ""
    feedback: str = ""
    correction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tool": self.tool,
            "arguments": to_jsonable(self.arguments),
            "result": to_jsonable(self.result),
            "user_input": self.user_input,
            "final_response": self.final_response,
            "feedback": self.feedback,
            "correction": self.correction,
            "metadata": to_jsonable(self.metadata),
        }


class EvolutionRecorder:
    """Append-only recorder for tool traces and exported learning batches."""

    def __init__(self, trace_dir: str = "./checkpoints/evolution"):
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        self.trace_path = os.path.join(trace_dir, "tool_traces.jsonl")

    def record_tool_call(
        self,
        tool: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        user_input: str = "",
        final_response: str = "",
        feedback: str = "",
        correction: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolTrace:
        trace = ToolTrace(
            timestamp=time.time(),
            tool=tool,
            arguments=arguments,
            result=result,
            user_input=user_input,
            final_response=final_response,
            feedback=feedback,
            correction=correction,
            metadata=metadata or {},
        )
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return trace

    def read_traces(self, limit: int = 0) -> List[dict]:
        if not os.path.exists(self.trace_path):
            return []
        traces: List[dict] = []
        with open(self.trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traces.append(json.loads(line))
        if limit > 0:
            return traces[-limit:]
        return traces

    def export_training_batch(
        self,
        cognitive_interface,
        out_dir: str,
        include_memory: bool = True,
    ) -> Dict[str, str]:
        """Export SFT/tool-use data from current interactions and memory."""
        os.makedirs(out_dir, exist_ok=True)

        paths = {
            "sft": os.path.join(out_dir, "sft_messages.jsonl"),
            "tool": os.path.join(out_dir, "tool_traces.jsonl"),
            "preference": os.path.join(out_dir, "preference_pairs.jsonl"),
        }
        if include_memory:
            paths["memory"] = os.path.join(out_dir, "memory_sft.jsonl")

        self._export_sft_messages(cognitive_interface, paths["sft"])
        self._export_tool_messages(paths["tool"])
        self._export_preference_pairs(paths["preference"])
        if include_memory:
            self._export_memory_messages(cognitive_interface, paths["memory"])

        manifest = {
            "created_at": time.time(),
            "files": paths,
            "format": {
                "sft_messages": "JSONL rows with {'messages': [{'role': ..., 'content': ...}]}",
                "tool_traces": "JSONL rows teaching the LLM to call PHYSMOL tools",
                "preference_pairs": "JSONL rows with chosen/rejected responses from feedback",
            },
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        paths["manifest"] = manifest_path
        return paths

    def _export_sft_messages(self, ci, path: str):
        rows: List[dict] = []

        for turn in getattr(ci, "_history", []):
            user = turn.get("user", "")
            response = turn.get("response", "")
            if user and response:
                rows.append({
                    "messages": [
                        {"role": "system", "content": _system_training_prompt()},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "source": "physmol_cognitive_history",
                        "intent": turn.get("intent", ""),
                    },
                })

        learner = getattr(ci, "learner", None)
        if learner is not None:
            for record in learner.get_history(limit=100000):
                user = record.get("user_input", "")
                response = record.get("system_response", "")
                if user and response:
                    rows.append({
                        "messages": [
                            {"role": "system", "content": _system_training_prompt()},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": response},
                        ],
                        "metadata": {
                            "source": "physmol_continuous_learner",
                            "intent": record.get("intent", ""),
                            "feedback": record.get("feedback", ""),
                        },
                    })

        _write_jsonl(path, rows)

    def _export_tool_messages(self, path: str):
        rows = []
        for trace in self.read_traces():
            user = trace.get("user_input") or _tool_instruction(trace)
            tool = trace.get("tool", "")
            args = trace.get("arguments", {})
            result = trace.get("result", {})
            final_response = trace.get("final_response") or _summarize_tool_result(result)
            rows.append({
                "messages": [
                    {"role": "system", "content": _system_training_prompt()},
                    {"role": "user", "content": user},
                    {
                        "role": "assistant",
                        "content": (
                            "I should call PHYSMOL tool "
                            f"`{tool}` with arguments: {json.dumps(args, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": final_response,
                    },
                ],
                "metadata": {
                    "source": "physmol_tool_trace",
                    "tool": tool,
                },
                "tool_result": result,
            })
        _write_jsonl(path, rows)

    def _export_preference_pairs(self, path: str):
        rows = []
        for trace in self.read_traces():
            if trace.get("feedback") != "bad" or not trace.get("correction"):
                continue
            rows.append({
                "prompt": trace.get("user_input", ""),
                "chosen": trace["correction"],
                "rejected": trace.get("final_response", ""),
                "metadata": {
                    "source": "physmol_feedback",
                    "tool": trace.get("tool", ""),
                },
            })
        _write_jsonl(path, rows)

    def _export_memory_messages(self, ci, path: str):
        rows = []
        memory = getattr(ci, "long_term_memory", None)
        if memory is None:
            _write_jsonl(path, rows)
            return
        for rec in memory.list_records():
            rows.append({
                "messages": [
                    {"role": "system", "content": _system_training_prompt()},
                    {
                        "role": "user",
                        "content": f"Remember this PHYSMOL knowledge: {rec.content}",
                    },
                    {
                        "role": "assistant",
                        "content": f"I will store it as {rec.memory_type} memory: {rec.content}",
                    },
                ],
                "metadata": {
                    "source": "physmol_long_term_memory",
                    "memory_id": rec.memory_id,
                    "memory_type": rec.memory_type,
                    "tags": rec.tags,
                },
            })
        _write_jsonl(path, rows)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_jsonl(paths: Iterable[str], out_path: str):
    rows = []
    for path in paths:
        if os.path.exists(path):
            rows.extend(load_jsonl(path))
    _write_jsonl(out_path, rows)


def _write_jsonl(path: str, rows: Iterable[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def _system_training_prompt() -> str:
    return (
        "You are a language shell for PHYSMOL. Use PHYSMOL tools for physics, "
        "memory, concept learning, counterfactuals, and self-improvement data. "
        "Answer naturally, but keep claims grounded in tool results."
    )


def _tool_instruction(trace: dict) -> str:
    tool = trace.get("tool", "physmol_query")
    args = json.dumps(trace.get("arguments", {}), ensure_ascii=False)
    return f"Use PHYSMOL tool `{tool}` with arguments {args}."


def _summarize_tool_result(result: dict) -> str:
    if "response" in result:
        return str(result["response"])
    if "message" in result:
        return str(result["message"])
    return json.dumps(to_jsonable(result), ensure_ascii=False)
