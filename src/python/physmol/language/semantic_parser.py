"""SemanticParser: Match text-encoded VSA vectors against the concept library.

Given a text query encoded as a VSA vector, the parser:
  1. Searches the recipe store for matching objects (resonance)
  2. Decomposes the query into attribute components
  3. Identifies the intent (question, command, explanation request)
  4. Activates relevant causal graph nodes for reasoning
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from ..vsa_store import RecipeStore, ConceptSynthesizer, AttributePrimitivePool
from .text_encoder import TextToVSA


# Intent classification keywords
INTENT_PATTERNS = {
    "question": [
        "what", "where", "when", "how", "why", "which", "who",
        "does", "do", "did", "is", "are", "was", "were", "can",
        "could", "would", "will", "should", "?"
    ],
    "command": [
        "put", "place", "move", "push", "pull", "lift", "drop",
        "stop", "start", "make", "create", "remove", "delete",
        "turn", "rotate", "flip", "stack", "arrange"
    ],
    "explanation": [
        "explain", "describe", "what is", "what are", "define",
        "tell me about", "how does", "how do", "why does",
        "meaning of", "concept of"
    ],
    "counterfactual": [
        "what if", "what would", "if", "suppose", "imagine",
        "instead", "changed", "different", "happen if", "如果", "假如"
    ],
    "abstract": [
        "fairness", "fair", "justice", "law", "democracy", "freedom",
        "equality", "rights", "slavery", "wrong", "公平", "正义", "法律",
        "民主", "自由", "平等", "权利", "奴隶制", "错误"
    ],
    "social": [
        "believe", "thinks", "think", "knows", "intend", "intention",
        "wants", "desire", "feel", "emotion", "false belief", "认为",
        "相信", "知道", "意图", "想要", "希望", "情绪", "觉得"
    ],
}


class SemanticParser:
    """Parse natural language queries against the VSA concept library.

    Workflow:
      1. Text -> VSA vector (via TextToVSA)
      2. Extract attribute hints from keywords
      3. Resonance search in recipe store
      4. Classify intent
      5. Return structured query for reasoning engine
    """

    def __init__(self, text_encoder: TextToVSA, recipe_store: RecipeStore):
        self.text_encoder = text_encoder
        self.recipe_store = recipe_store
        self.synthesizer = ConceptSynthesizer(recipe_store)
        self.primitives = recipe_store.primitives

        # Build keyword -> attribute ID mapping
        self._keyword_to_attr = self._build_keyword_map()

    def _build_keyword_map(self) -> Dict[str, List[str]]:
        """Map common words to attribute IDs."""
        kw_map = {}

        # Shape words
        for shape in ["sphere", "cube", "cylinder", "capsule", "cone"]:
            kw_map[shape] = [f"shape_{shape}"]
        kw_map["ball"] = ["shape_sphere"]
        kw_map["球"] = ["shape_sphere"]
        kw_map["block"] = ["shape_cube"]
        kw_map["box"] = ["shape_cube"]
        kw_map["方块"] = ["shape_cube"]
        kw_map["立方体"] = ["shape_cube"]
        kw_map["rod"] = ["shape_cylinder"]
        kw_map["tube"] = ["shape_cylinder"]

        # Color words
        for color in ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]:
            kw_map[color] = [f"color_{color}"]
        kw_map["红色"] = ["color_red"]
        kw_map["蓝色"] = ["color_blue"]

        # Material words
        for mat in ["metal", "wood", "plastic", "rubber", "glass", "stone", "fabric"]:
            kw_map[mat] = [f"material_{mat}"]
        kw_map["steel"] = ["material_metal"]
        kw_map["iron"] = ["material_metal"]
        kw_map["paper"] = ["material_fabric"]

        # Mass words
        kw_map["heavy"] = ["mass_heavy"]
        kw_map["light"] = ["mass_light"]
        kw_map["重"] = ["mass_heavy"]
        kw_map["轻"] = ["mass_light"]
        kw_map["big"] = ["mass_heavy"]
        kw_map["small"] = ["mass_light"]

        # Elasticity words
        kw_map["elastic"] = ["elasticity_elastic"]
        kw_map["弹性"] = ["elasticity_elastic"]
        kw_map["bouncy"] = ["elasticity_elastic"]
        kw_map["rigid"] = ["elasticity_rigid"]
        kw_map["stiff"] = ["elasticity_stiff"]
        kw_map["soft"] = ["elasticity_soft"]
        kw_map["hard"] = ["elasticity_rigid"]

        # Texture words
        for tex in ["smooth", "rough"]:
            kw_map[tex] = [f"texture_{tex}"]

        return kw_map

    def extract_attribute_hints(self, tokens: List[str]) -> List[str]:
        """Extract attribute IDs from keyword matching."""
        attr_ids = []
        for token in tokens:
            if token in self._keyword_to_attr:
                attr_ids.extend(self._keyword_to_attr[token])
        return list(set(attr_ids))

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of a text query.

        Returns: (intent_type, confidence)
        intent_type: "question", "command", "explanation", "counterfactual", "unknown"
        """
        text_lower = text.lower().strip()
        tokens = set(self.text_encoder.tokenize(text_lower))

        scores = {}
        for intent, keywords in INTENT_PATTERNS.items():
            score = 0
            for kw in keywords:
                if kw in text_lower:
                    # Multi-word keywords get higher weight
                    score += len(kw.split())
            scores[intent] = score

        if not scores or max(scores.values()) == 0:
            return "unknown", 0.0

        best_intent = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_intent] / total if total > 0 else 0.0

        return best_intent, confidence

    def find_matching_objects(self, text: str, top_k: int = 5
                              ) -> List[Tuple[str, float]]:
        """Find objects in the recipe store that match the text query.

        Strategy: combine VSA resonance with keyword attribute matching.
        """
        # Method 1: VSA resonance
        query_vec = self.text_encoder.encode(text)
        vsa_matches = self.recipe_store.resonate(query_vec, top_k=top_k * 2)

        # Method 2: keyword attribute matching
        tokens = self.text_encoder.tokenize(text)
        attr_hints = self.extract_attribute_hints(tokens)
        kw_matches = self.recipe_store.find_by_attributes(attr_hints, top_k=top_k * 2)

        # Merge scores (weighted combination)
        scores: Dict[str, float] = {}
        for obj_id, sim in vsa_matches:
            scores[obj_id] = scores.get(obj_id, 0) + 0.5 * sim
        for obj_id, score in kw_matches:
            scores[obj_id] = scores.get(obj_id, 0) + 0.5 * score

        # Sort by combined score
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def parse_query(self, text: str) -> dict:
        """Parse a natural language query into a structured representation.

        Returns:
            {
                "text": original text,
                "tokens": tokenized words,
                "intent": "question" | "command" | "explanation" | "counterfactual",
                "intent_confidence": float,
                "vsa_vector": numpy array,
                "attribute_hints": [attr_id, ...],
                "matching_objects": [(obj_id, score), ...],
                "decomposition": {category: (name, sim)},
            }
        """
        tokens = self.text_encoder.tokenize(text)
        vsa_vec = self.text_encoder.encode(text)
        intent, confidence = self.classify_intent(text)
        attr_hints = self.extract_attribute_hints(tokens)
        matching = self.find_matching_objects(text)
        decomposition = self.recipe_store.decompose(vsa_vec)

        return {
            "text": text,
            "tokens": tokens,
            "intent": intent,
            "intent_confidence": confidence,
            "vsa_vector": vsa_vec,
            "attribute_hints": attr_hints,
            "matching_objects": matching,
            "decomposition": decomposition,
        }

    def activate_causal_nodes(self, parsed_query: dict) -> List[str]:
        """Identify which causal graph nodes should be activated.

        Based on matched objects and detected attributes, return a list
        of node IDs that the reasoning engine should focus on.
        """
        nodes = []

        # Activate nodes for matched objects
        for obj_id, score in parsed_query.get("matching_objects", []):
            if score > 0.3:
                nodes.append(f"obj:{obj_id}")

        # Activate nodes for detected attributes
        for attr_id in parsed_query.get("attribute_hints", []):
            nodes.append(f"attr:{attr_id}")

        # Activate physics concept nodes based on keywords
        tokens = parsed_query.get("tokens", [])
        physics_concepts = {
            "fall": "gravity", "drop": "gravity", "roll": "rolling",
            "bounce": "elasticity", "slide": "friction", "collide": "collision",
            "push": "force", "pull": "force", "lift": "force",
            "force": "force", "energy": "energy", "momentum": "momentum",
            "speed": "velocity", "fast": "velocity", "slow": "velocity",
            "heavy": "mass", "light": "mass",
        }
        for token in tokens:
            if token in physics_concepts:
                nodes.append(f"physics:{physics_concepts[token]}")

        return list(set(nodes))
