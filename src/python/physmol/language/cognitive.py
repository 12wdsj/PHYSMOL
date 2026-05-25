"""CognitiveInterface: Unified API for PHYSMOL language interaction.

This is the main entry point for natural language queries.
It orchestrates: text encoding -> semantic parsing -> reasoning -> response.

Usage:
    from physmol.language import CognitiveInterface

    ci = CognitiveInterface(vsa_dim=10000)

    # Register objects from physical exploration
    ci.register_object("ball_1", ["shape_sphere", "color_red", "material_rubber", "elasticity_elastic"])

    # Ask questions
    response = ci.query("What happens if I drop the red ball?")

    # Get explanations
    response = ci.query("Explain elasticity")

    # Execute commands
    response = ci.query("Push the cube to the top of the ramp")
"""

from typing import Dict, List, Optional, Any

from ..vsa_store import AttributePrimitivePool, RecipeStore
from .text_encoder import TextToVSA
from .semantic_parser import SemanticParser
from .reasoning import ReasoningEngine
from .responder import Responder
from .conversation import DialogueState
from .theory_of_mind import TheoryOfMindModel
from .abstract_tasks import AbstractTaskReasoner
from ..motivation import IntrinsicMotivationSystem
from ..long_term_memory import LongTermMemory
from ..transfer import CrossDomainTransferEngine


class CognitiveInterface:
    """Unified natural language interface for PHYSMOL.

    Orchestrates the full pipeline:
      User text -> TextToVSA -> SemanticParser -> ReasoningEngine -> Responder -> Response
    """

    def __init__(self, vsa_dim: int = 10000, seed: int = 42,
                 causal_graph=None, lgnn=None):
        # VSA infrastructure
        self.primitives = AttributePrimitivePool(vsa_dim, seed)
        self.recipe_store = RecipeStore(self.primitives)

        # Language components
        self.text_encoder = TextToVSA(vsa_dim, seed)
        self.semantic_parser = SemanticParser(self.text_encoder, self.recipe_store)
        self.reasoning_engine = ReasoningEngine(
            self.recipe_store, causal_graph, lgnn)
        self.responder = Responder()
        self.dialogue = DialogueState()
        self.theory_of_mind = TheoryOfMindModel()
        self.motivation = IntrinsicMotivationSystem()
        self.long_term_memory = LongTermMemory(self.text_encoder)
        self.transfer_engine = CrossDomainTransferEngine()
        self.abstract_task_reasoner = AbstractTaskReasoner()

        # Conversation history
        self._history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Object management (interface to physical world)
    # ------------------------------------------------------------------

    def register_object(self, object_id: str, attribute_ids: List[str]):
        """Register an object discovered through physical exploration.

        Args:
            object_id: unique identifier (e.g. "ball_1", "red_sphere_3")
            attribute_ids: list of attribute IDs from the primitive pool
                e.g. ["shape_sphere", "color_red", "material_rubber"]
        """
        self.recipe_store.register_recipe(object_id, attribute_ids)

    def remove_object(self, object_id: str):
        """Remove an object from the knowledge base."""
        self.recipe_store.remove_recipe(object_id)

    def list_objects(self) -> List[str]:
        """List all known objects."""
        return self.recipe_store.list_objects()

    def get_object_attributes(self, object_id: str) -> Dict[str, str]:
        """Get human-readable attributes of an object."""
        recipe = self.recipe_store.get_recipe(object_id)
        if recipe is None:
            return {}
        attrs = {}
        for attr_id in recipe:
            cat, name = self.primitives.resolve_id(attr_id)
            if cat:
                attrs[cat] = name
        return attrs

    # ------------------------------------------------------------------
    # Natural language query
    # ------------------------------------------------------------------

    def query(self, text: str) -> str:
        """Process a natural language query and return a response.

        Full pipeline:
          1. Parse text -> intent + attributes + matching objects
          2. Route to appropriate reasoning method
          3. Generate natural language response

        Args:
            text: natural language input

        Returns: response string
        """
        # Step 1: Parse
        parsed = self.semantic_parser.parse_query(text)

        conversation = self.dialogue.conversational_response(text)
        if conversation is not None:
            response = self.responder.respond(parsed, conversation)
            self._record_turn(text, "conversation", response, parsed)
            return response

        # Step 2: Reason based on intent
        intent = parsed["intent"]

        tom_update = self.theory_of_mind.update_from_text(text)
        if tom_update is not None and not self._looks_like_question(text):
            parsed["intent"] = "social"
            response = self.responder.respond(parsed, tom_update)
            self._record_turn(text, "social", response, parsed)
            return response

        memory_result = self._handle_memory_request(text, parsed)

        if self.theory_of_mind.can_answer(text):
            parsed["intent"] = "social"
            result = self.theory_of_mind.answer(text)
        elif memory_result is not None:
            parsed["intent"] = "memory"
            result = memory_result
        elif self._looks_like_transfer_request(text):
            parsed["intent"] = "transfer"
            result = self._handle_transfer_request(text)
        elif self._looks_like_abstract_task(text):
            parsed["intent"] = "abstract_task"
            result = self.abstract_task_reasoner.reason(text)
        elif self.reasoning_engine.abstract_reasoner.has_signal(text):
            parsed["intent"] = "abstract"
            result = self.reasoning_engine.reason_abstract(text)
        elif intent == "question":
            result = self._handle_question(parsed)
        elif intent == "command":
            result = self._handle_command(parsed)
        elif intent == "explanation":
            result = self._handle_explanation(parsed)
        elif intent == "counterfactual":
            result = self._handle_counterfactual(parsed)
        else:
            result = {"error": "unknown intent"}

        # Step 3: Generate response
        response = self.responder.respond(parsed, result)

        # Record in history
        self._record_turn(text, parsed["intent"], response, parsed)

        return response

    def chat(self, text: str) -> str:
        """Alias for query(), useful for interactive shells."""
        return self.query(text)

    def _handle_question(self, parsed: dict) -> dict:
        """Handle a question-type query."""
        tokens = parsed["tokens"]

        # Check if it's asking about a specific object
        if parsed["matching_objects"]:
            best_obj, best_score = parsed["matching_objects"][0]

            # Check for physics-related question
            physics_keywords = {"what happen", "what if", "fall", "drop", "roll",
                                "bounce", "move", "fast", "slow", "heavy"}
            is_physics_q = any(kw in " ".join(tokens) for kw in physics_keywords)

            if is_physics_q:
                # Use reasoning engine for physics prediction
                context = {}
                if "drop" in tokens or "fall" in tokens:
                    context["action"] = "drop"
                elif "push" in tokens:
                    context["action"] = "push"
                elif "roll" in tokens:
                    context["action"] = "roll"
                elif "collide" in tokens or "hit" in tokens:
                    context["action"] = "collide"

                prediction = self.reasoning_engine.predict(
                    parsed["text"], context)
                return prediction
            else:
                # Just return info about the matched object
                attrs = self.get_object_attributes(best_obj)
                return {
                    "prediction": f"I found '{best_obj}' with attributes: {attrs}",
                    "matched_objects": parsed["matching_objects"],
                    "applicable_rules": [],
                }

        # No object match -- try concept explanation
        for token in tokens:
            if token in self.reasoning_engine._physics_rules:
                return self.reasoning_engine.explain_concept(token)

        return {"prediction": "I'm not sure what you're asking about. Could you be more specific?"}

    def _handle_command(self, parsed: dict) -> dict:
        """Handle a command-type query."""
        return self.reasoning_engine.plan_action(parsed["text"])

    def _handle_explanation(self, parsed: dict) -> dict:
        """Handle an explanation request."""
        tokens = parsed["tokens"]

        # Find the concept to explain
        concept_keywords = ["elasticity", "gravity", "friction", "momentum",
                            "energy", "inertia", "force", "bounce", "collision"]
        for token in tokens:
            if token in concept_keywords:
                return self.reasoning_engine.explain_concept(token)

        # Check for compound concepts
        text_lower = " ".join(tokens)
        for concept in concept_keywords:
            if concept in text_lower:
                return self.reasoning_engine.explain_concept(concept)

        return {"concept": "unknown", "explanation": {
            "definition": "I'm not sure which concept you'd like me to explain.",
            "physics": "Try asking about: elasticity, gravity, friction, momentum, energy, inertia, or force.",
            "examples": [],
        }}

    def _handle_counterfactual(self, parsed: dict) -> dict:
        """Handle a counterfactual question."""
        tokens = parsed["tokens"]
        text = parsed["text"]

        # Extract subject and change
        subject = "the object"
        change = "a property"

        # Find subject (object keywords)
        for token in tokens:
            if token in ["ball", "cube", "block", "sphere", "cylinder", "box"]:
                subject = token
                break

        # Find what changed
        change_keywords = {
            "heavy": "mass (heavier)",
            "light": "mass (lighter)",
            "big": "size (larger)",
            "small": "size (smaller)",
            "elastic": "elasticity (more elastic)",
            "rigid": "elasticity (rigid)",
            "smooth": "friction (less friction)",
            "rough": "friction (more friction)",
            "strong": "gravity (stronger)",
            "weak": "gravity (weaker)",
        }
        for token in tokens:
            if token in change_keywords:
                change = change_keywords[token]
                break

        # Check for "what if" pattern
        if "what if" in text.lower():
            # Extract the part after "what if"
            idx = text.lower().index("what if") + len("what if")
            change = text[idx:].strip().rstrip("?")

        return self.reasoning_engine.counterfactual(subject, change)

    # ------------------------------------------------------------------
    # Social cognition
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str):
        """Create or retrieve an agent model for theory-of-mind reasoning."""
        return self.theory_of_mind.ensure_agent(agent_id)

    def observe_agent(self, agent_id: str, **mental_state) -> dict:
        """Update another agent's modeled beliefs, intentions, desires, or emotions."""
        return self.theory_of_mind.observe(agent_id, **mental_state)

    # ------------------------------------------------------------------
    # Long-term memory and transfer
    # ------------------------------------------------------------------

    def remember_fact(self, subject: str, predicate: str, obj: str,
                      evidence: Optional[str] = None, confidence: float = 0.8):
        return self.long_term_memory.add_fact(
            subject, predicate, obj, evidence=evidence, confidence=confidence)

    def remember_episode(self, content: str, **kwargs):
        return self.long_term_memory.add_episode(content, **kwargs)

    def recall(self, query: str, top_k: int = 5):
        return self.long_term_memory.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return list(self._history)

    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()
        self.dialogue = DialogueState()

    def get_status(self) -> dict:
        """Get system status."""
        return {
            "vsa_dim": self.primitives.vsa_dim,
            "vocabulary_size": self.text_encoder.lexicon.vocabulary_size,
            "num_categories": len(self.primitives.list_categories()),
            "num_primitives": len(self.primitives),
            "num_objects": len(self.recipe_store),
            "num_conversations": len(self._history),
            "num_agents": len(self.theory_of_mind.agents),
            "dialogue": self.dialogue.context(),
            "motivation": self.motivation.summary(),
            "long_term_memory": self.long_term_memory.stats(),
            "transfer_domains": list(self.transfer_engine.domains.keys()),
        }

    def record_learning_signal(self, context_key: str, prediction_error: float,
                               uncertainty: Optional[float] = None) -> dict:
        """Expose intrinsic motivation updates to training loops."""
        return self.motivation.record_outcome(context_key, prediction_error, uncertainty)

    def _record_turn(self, text: str, intent: str, response: str, parsed: dict):
        self._history.append({
            "user": text,
            "intent": intent,
            "response": response,
        })
        self.dialogue.record(text, intent, response, parsed)
        if intent not in {"memory"}:
            self.long_term_memory.add_episode(
                content=text,
                outcome=response,
                tags=["conversation", intent],
            )

    def _looks_like_question(self, text: str) -> bool:
        lower = text.lower().strip()
        return (
            lower.endswith("?")
            or text.strip().endswith("？")
            or any(lower.startswith(prefix) for prefix in (
                "what", "why", "how", "where", "when", "does", "do", "is",
                "are", "can", "could", "would", "should",
            ))
            or any(marker in text for marker in ("什么", "为什么", "怎么", "吗", "？"))
        )

    def _handle_memory_request(self, text: str, parsed: dict) -> Optional[dict]:
        lower = text.lower().strip()
        if lower.startswith("remember that "):
            content = text[len("remember that "):].strip()
            rec = self.long_term_memory.add_episode(
                content=content, tags=["user_taught", "language"])
            return {
                "kind": "memory",
                "action": "stored",
                "records": [rec.to_dict()],
            }
        if text.startswith("记住"):
            content = text[len("记住"):].strip(" ：:")
            rec = self.long_term_memory.add_episode(
                content=content, tags=["user_taught", "language"])
            return {
                "kind": "memory",
                "action": "stored",
                "records": [rec.to_dict()],
            }
        if "what do you remember" in lower or "recall" in lower or "回忆" in text or "记得" in text:
            query = text
            for prefix in ["what do you remember about", "recall"]:
                if prefix in lower:
                    query = text[lower.index(prefix) + len(prefix):].strip(" ?")
                    break
            records = self.long_term_memory.retrieve(query, top_k=5)
            return {
                "kind": "memory",
                "action": "retrieved",
                "query": query,
                "records": [
                    {"memory_id": rec.memory_id, "content": rec.content,
                     "memory_type": rec.memory_type, "score": score,
                     "tags": rec.tags}
                    for rec, score in records
                ],
            }
        return None

    def _looks_like_transfer_request(self, text: str) -> bool:
        lower = text.lower()
        return any(cue in lower for cue in ["transfer", "chess", "blocks"]) or \
            any(cue in text for cue in ["迁移", "下棋", "积木"])

    def _handle_transfer_request(self, text: str) -> dict:
        source = "blocks" if ("block" in text.lower() or "积木" in text) else "source"
        target = "chess" if ("chess" in text.lower() or "下棋" in text or "棋" in text) else "target"
        patterns = self.transfer_engine.lift_experience_to_patterns(text)
        if not patterns and source == "blocks" and target == "chess":
            patterns = ["spatial_support", "blocking_constraint", "action_changes_state"]
        return self.transfer_engine.propose_transfer(source, target, patterns)

    def _looks_like_abstract_task(self, text: str) -> bool:
        lower = text.lower()
        cues = [
            "prove", "theorem", "math", "legal reasoning", "moral judgment",
            "ethical", "court", "contract",
        ]
        zh_cues = ["证明", "定理", "数学", "法律推理", "道德判断", "伦理", "法院", "合同"]
        return any(cue in lower for cue in cues) or any(cue in text for cue in zh_cues)
