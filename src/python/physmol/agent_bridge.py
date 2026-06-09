"""Agent and small-LLM bridge for PHYSMOL.

The bridge exposes PHYSMOL as a set of stable tools.  A small LLM can use those
tools for grounded cognition while the PHYSMOL side keeps learning concepts,
memory, and feedback traces.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .evolution import EvolutionRecorder, to_jsonable
from .language.cognitive import CognitiveInterface


DEFAULT_SYSTEM_PROMPT = """You are PHYSMOL's voice. Answer the user's question using ONLY the JSON data below.
Rules:
- Be concise: 1-3 sentences max.
- Do NOT dump raw JSON or list all attributes.
- Focus on the key concept, formula, or answer the user asked for.
- If the data says "unknown", say you don't know that concept yet.
- Speak naturally. No "Based on the JSON..." or "The tool says...".
"""


CONCEPT_ALIASES = {
    # Chinese physical concepts
    "重力": "gravity",
    "引力": "gravity",
    "动量": "momentum",
    "能量": "energy",
    "摩擦": "friction",
    "摩擦力": "friction",
    "弹性": "elasticity",
    "惯性": "inertia",
    "力": "force",
    "动量守恒": "momentum",
    "动量守恒定律": "momentum",
    "能量守恒": "energy",
    "能量守恒定律": "energy",
    "牛顿第一定律": "inertia",
    "牛顿第二定律": "force",
    "牛顿第三定律": "force",
    "牛顿定律": "force",
    "万有引力": "gravity",
    "弹性碰撞": "elasticity",
    "非弹性碰撞": "collision",
    "重力势能": "energy",
    "动能": "energy",
    "势能": "energy",
    "加速度": "force",
    "速度": "momentum",
    "质量": "inertia",
    "碰撞": "collision",
    "热力学": "energy",
    "守恒定律": "energy",
    # Chinese code concepts
    "递归": "recursion",
    "快速排序": "quicksort",
    "快排": "quicksort",
    "归并排序": "merge sort",
    "堆排序": "heap sort",
    "二分查找": "binary search",
    "广度优先搜索": "bfs",
    "深度优先搜索": "dfs",
    "动态规划": "dynamic programming",
    "链表": "linked list",
    "栈": "stack",
    "队列": "queue",
    "图": "graph",
    "二叉树": "binary tree",
    "哈希表": "hash map",
    "冒泡排序": "quicksort",
    "排序算法": "quicksort",
    "搜索算法": "binary search",
}


CODE_CONCEPTS = {
    "quicksort", "merge sort", "heap sort", "binary search", "bfs", "dfs",
    "dynamic programming", "linked list", "stack", "queue", "graph",
    "recursion", "dijkstra", "binary tree", "hash map", "lru cache",
}


@dataclass
class SmallLLMClient:
    """HTTP client for Ollama /api/generate endpoint (default) or OpenAI-compatible APIs.

    Supported providers:
      - ollama: Ollama native /api/generate endpoint (default, best for thinking models)
      - openai: OpenAI-compatible /v1/chat/completions endpoint
    """

    provider: str = "ollama"
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    model: str = "deepseek-r1:1.5b"
    api_key: str = ""
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "SmallLLMClient":
        """Create client from env vars, falling back to Ollama defaults."""
        endpoint = os.environ.get("PHYSMOL_LLM_ENDPOINT", "").strip()
        model = os.environ.get("PHYSMOL_LLM_MODEL", "").strip()
        provider = os.environ.get("PHYSMOL_LLM_PROVIDER", "").strip()
        return cls(
            provider=provider or "ollama",
            endpoint=endpoint or "http://127.0.0.1:11434/api/generate",
            model=model or "deepseek-r1:1.5b",
            api_key=os.environ.get("PHYSMOL_LLM_API_KEY", ""),
            timeout=120.0,
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        if self.provider == "ollama":
            return self._chat_ollama(messages, temperature)
        return self._chat_openai_compatible(messages, temperature)

    def _chat_openai_compatible(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def _chat_ollama(self, messages: List[Dict[str, str]], temperature: float) -> str:
        import re as _re
        # Convert OpenAI-style messages to Ollama Generate API format
        system_parts = []
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)
        payload = {
            "model": self.model,
            "prompt": "\n".join(user_parts),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512},
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        # Generate API returns {"response": "...", "thinking": "..."}
        # thinking model (deepseek-r1, qwen3) puts reasoning in "thinking"
        # and final answer in "response".  Fall back to thinking if response is empty.
        content = body.get("response", "").strip()
        if not content:
            content = body.get("thinking", "").strip()
        # Strip <think>...</think> tags if still present
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
        return content


class PhysmolAgentBridge:
    """Stable tool bridge used by agents, MCP clients, and local LLM shells."""

    def __init__(
        self,
        vsa_dim: int = 4096,
        broca_checkpoint: str = "./checkpoints/broca/model",
        learning_dir: str = "./checkpoints/learning",
        trace_dir: str = "./checkpoints/evolution",
        llm_client: Optional[SmallLLMClient] = None,
    ):
        self.ci = CognitiveInterface(vsa_dim=vsa_dim)
        self.learning_dir = learning_dir
        self.recorder = EvolutionRecorder(trace_dir)
        self.llm_client = llm_client if llm_client is not None else SmallLLMClient.from_env()

        if broca_checkpoint and os.path.exists(broca_checkpoint):
            try:
                self.ci.broca.load(broca_checkpoint)
            except Exception:
                # Broca is optional; the bridge still works through normal PHYSMOL tools.
                pass

        if learning_dir and os.path.exists(learning_dir):
            try:
                self.ci.learner.load(learning_dir)
            except Exception:
                pass

        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "physmol_query": self._tool_query,
            "explain_concept": self._tool_explain_concept,
            "simulate_counterfactual": self._tool_counterfactual,
            "teach_concept": self._tool_teach_concept,
            "remember_fact": self._tool_remember_fact,
            "recall_memory": self._tool_recall_memory,
            "record_feedback": self._tool_record_feedback,
            "export_training_batch": self._tool_export_training_batch,
            "save_learning_state": self._tool_save_learning_state,
            "load_learning_state": self._tool_load_learning_state,
            "status": self._tool_status,
        }

    def list_tools(self) -> List[dict]:
        """Return tool descriptors with JSON schemas."""
        return [
            {
                "name": "physmol_query",
                "description": "Ask PHYSMOL a natural-language question. Use for general PHYSMOL reasoning.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "learn": {
                            "type": "boolean",
                            "description": "If true, route through ContinuousLearner.",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "explain_concept",
                "description": "Explain a physical, algorithmic, or learned concept using PHYSMOL.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"concept": {"type": "string"}},
                    "required": ["concept"],
                },
            },
            {
                "name": "simulate_counterfactual",
                "description": "Run a PHYSMOL counterfactual over an object/property change.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "change": {"type": "string"},
                    },
                    "required": ["subject", "change"],
                },
            },
            {
                "name": "teach_concept",
                "description": "Teach PHYSMOL a new concept and store it in concept memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "category": {"type": "string"},
                        "definition": {"type": "string"},
                        "examples": {"type": "array", "items": {"type": "string"}},
                        "related": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["term"],
                },
            },
            {
                "name": "remember_fact",
                "description": "Store a semantic fact in PHYSMOL long-term memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "evidence": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
            {
                "name": "recall_memory",
                "description": "Retrieve PHYSMOL long-term memories relevant to a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "record_feedback",
                "description": "Record user feedback on the latest interaction.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "feedback": {"type": "string", "enum": ["good", "bad"]},
                        "correction": {"type": "string"},
                    },
                    "required": ["feedback"],
                },
            },
            {
                "name": "export_training_batch",
                "description": "Export SFT/tool/preference JSONL files for small-LLM fine-tuning.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "out_dir": {"type": "string"},
                        "include_memory": {"type": "boolean"},
                    },
                    "required": ["out_dir"],
                },
            },
            {
                "name": "save_learning_state",
                "description": "Persist ContinuousLearner, Broca, and long-term memory state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "name": "load_learning_state",
                "description": "Load ContinuousLearner/Broca state from disk.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "name": "status",
                "description": "Return PHYSMOL, learner, memory, and bridge status.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        user_input: str = "",
        record: bool = True,
    ) -> Dict[str, Any]:
        arguments = arguments or {}
        if name not in self._tools:
            raise ValueError(f"Unknown PHYSMOL tool: {name}")
        result = self._tools[name](arguments)
        if record:
            self.recorder.record_tool_call(
                tool=name,
                arguments=arguments,
                result=result,
                user_input=user_input,
                final_response=result.get("response", ""),
            )
        return result

    def chat(self, text: str, use_llm: bool = True, learn: bool = True) -> str:
        """Ask PHYSMOL, then optionally let a small LLM verbalize the result."""
        tool_result = self.call_tool(
            "physmol_query",
            {"text": text, "learn": learn},
            user_input=text,
            record=True,
        )
        base_response = tool_result.get("response", "")

        if not use_llm or self.llm_client is None:
            return base_response

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"User question:\n{text}\n\n"
                    "PHYSMOL tool result as JSON:\n"
                    f"{json.dumps(to_jsonable(tool_result), ensure_ascii=False, indent=2)}\n\n"
                    "Answer naturally in the user's language. Keep it concise."
                ),
            },
        ]
        try:
            response = self.llm_client.chat(messages)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, OSError) as exc:
            response = f"{base_response}"

        self.recorder.record_tool_call(
            tool="llm_verbalize",
            arguments={"text": text, "base_response": base_response},
            result={"response": response, "tool_result": tool_result},
            user_input=text,
            final_response=response,
        )
        return response

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = str(args.get("text", ""))
        learn = bool(args.get("learn", False))
        response = self.ci.learner.interact(text) if learn else self.ci.query(text)
        parsed = self.ci.semantic_parser.parse_query(text)
        return {
            "response": response,
            "parsed": _compact_parsed(parsed),
            "learner": self.ci.learner.get_stats(),
        }

    def _tool_explain_concept(self, args: Dict[str, Any]) -> Dict[str, Any]:
        concept = canonicalize_concept(str(args.get("concept", "")))
        if concept in CODE_CONCEPTS:
            result = self.ci.reasoning_engine.explain_code_concept(concept)
        else:
            result = self.ci.reasoning_engine.explain_concept(concept)
        return {
            "concept": concept,
            "result": result,
            "response": self.ci.generator.generate_from_reasoning(
                {"intent": "explanation", "tokens": [concept]},
                result,
            ),
        }

    def _tool_counterfactual(self, args: Dict[str, Any]) -> Dict[str, Any]:
        subject = str(args.get("subject", "object"))
        change = _canonicalize_change(str(args.get("change", "")))
        result = self.ci.reasoning_engine.counterfactual(subject, change)
        return {
            "result": result,
            "response": self.ci.generator.generate_from_reasoning(
                {"intent": "counterfactual", "tokens": []},
                result,
            ),
        }

    def _tool_teach_concept(self, args: Dict[str, Any]) -> Dict[str, Any]:
        concept = self.ci.teach_concept(
            term=str(args.get("term", "")),
            category=str(args.get("category", "")),
            definition=str(args.get("definition", "")),
            examples=list(args.get("examples", []) or []),
            related=list(args.get("related", []) or []),
        )
        self.ci.long_term_memory.add_fact(
            subject=concept.term,
            predicate="definition",
            obj=concept.definition or f"{concept.category} concept",
            evidence="user_teaching",
            confidence=0.9,
            tags=["learned_concept"],
        )
        return {
            "concept": concept.to_dict() if hasattr(concept, "to_dict") else to_jsonable(concept),
            "response": f"Learned concept `{concept.term}` in category `{concept.category}`.",
        }

    def _tool_remember_fact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        rec = self.ci.remember_fact(
            subject=str(args.get("subject", "")),
            predicate=str(args.get("predicate", "")),
            obj=str(args.get("object", "")),
            evidence=str(args.get("evidence", "")) or None,
            confidence=float(args.get("confidence", 0.8)),
        )
        return {
            "memory": rec.to_dict(),
            "response": f"Stored fact memory `{rec.memory_id}`.",
        }

    def _tool_recall_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query", ""))
        top_k = int(args.get("top_k", 5))
        results = self.ci.recall(query, top_k=top_k)
        records = [
            {"record": rec.to_dict(), "score": score}
            for rec, score in results
        ]
        return {
            "records": records,
            "response": _format_memory_records(records),
        }

    def _tool_record_feedback(self, args: Dict[str, Any]) -> Dict[str, Any]:
        feedback = str(args.get("feedback", "")).strip().lower()
        correction = str(args.get("correction", ""))
        if feedback not in {"good", "bad"}:
            raise ValueError("feedback must be 'good' or 'bad'")
        self.ci.learner.record_feedback(feedback, correction)
        self.recorder.record_tool_call(
            tool="record_feedback",
            arguments=args,
            result={"feedback": feedback, "correction": correction},
            feedback=feedback,
            correction=correction,
        )
        return {
            "feedback": feedback,
            "correction": correction,
            "learner": self.ci.learner.get_stats(),
            "response": "Feedback recorded.",
        }

    def _tool_export_training_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        out_dir = str(args.get("out_dir", "./checkpoints/evolution/export"))
        include_memory = bool(args.get("include_memory", True))
        paths = self.recorder.export_training_batch(
            self.ci,
            out_dir=out_dir,
            include_memory=include_memory,
        )
        return {
            "paths": paths,
            "response": f"Exported training batch to {out_dir}.",
        }

    def _tool_save_learning_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = str(args.get("path") or self.learning_dir)
        os.makedirs(path, exist_ok=True)
        self.ci.learner.save(path)
        self.ci.long_term_memory.save_json(os.path.join(path, "long_term_memory.json"))
        return {"path": path, "response": f"Saved PHYSMOL learning state to {path}."}

    def _tool_load_learning_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        path = str(args.get("path") or self.learning_dir)
        self.ci.learner.load(path)
        memory_path = os.path.join(path, "long_term_memory.json")
        if os.path.exists(memory_path):
            self.ci.long_term_memory.load_json(memory_path)
        return {"path": path, "response": f"Loaded PHYSMOL learning state from {path}."}

    def _tool_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": self.ci.get_status(),
            "learner": self.ci.learner.get_stats(),
            "llm": {
                "configured": self.llm_client is not None,
                "provider": getattr(self.llm_client, "provider", ""),
                "model": getattr(self.llm_client, "model", ""),
                "endpoint": getattr(self.llm_client, "endpoint", ""),
            },
            "tools": [tool["name"] for tool in self.list_tools()],
        }


def canonicalize_concept(concept: str) -> str:
    raw = concept.strip()
    return CONCEPT_ALIASES.get(raw, raw.lower())


def _canonicalize_change(change: str) -> str:
    aliases = {
        "更重": "mass heavier",
        "更轻": "mass lighter",
        "质量翻倍": "mass double",
        "重力更强": "gravity stronger",
        "重力更弱": "gravity weaker",
        "没有摩擦": "friction zero",
        "摩擦更大": "friction more",
        "摩擦更小": "friction less",
        "更有弹性": "elasticity more_elastic",
        "没有弹性": "elasticity rigid",
    }
    return aliases.get(change.strip(), change)


def _compact_parsed(parsed: dict) -> dict:
    compact = {}
    for key in ["text", "tokens", "intent", "intent_confidence", "attribute_hints", "matching_objects", "decomposition"]:
        if key in parsed:
            compact[key] = to_jsonable(parsed[key])
    return compact


def _format_memory_records(records: List[dict]) -> str:
    if not records:
        return "No matching memory found."
    lines = []
    for item in records:
        rec = item["record"]
        lines.append(f"- [{rec.get('memory_type')}] {rec.get('content')} (score={item['score']:.3f})")
    return "\n".join(lines)
