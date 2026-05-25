"""PHYSMOL VSA Recipe Store -- Relationship Pattern Memory.

Core philosophy (MUST follow):
  VSA is NOT a database. It does NOT store raw object states (position, velocity,
  color values, etc.). It is a RELATIONSHIP PATTERN STORE that only stores:
    1. Global shared attribute primitive vectors (e.g. "sphere", "red", "elastic")
    2. Object recipes (each object = list of attribute IDs, NOT values)

Data flow:
  Physical simulator (WorldState) -> multimodal encoder -> extract attribute
  CATEGORIES (not continuous values) -> lookup primitive IDs from attribute
  pool -> VSA only stores the ID combination.

  Example: ball at (1,2,3), velocity (5,0,0), color #FF8C00 ->
  VSA stores: {id: "ball_123", recipe: ["shape_sphere", "color_orange", "material_rubber"]}

Prohibited:
  - Storing WorldState directly into VSA
  - Storing per-instance raw data vectors
  - Storing continuous values (position, velocity, color codes) as VSA content
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set


class AttributePrimitivePool:
    """Global shared pool of attribute primitive vectors.

    Each attribute category (shape, color, material, ...) has a set of
    randomly generated bipolar vectors. These are FIXED for the lifetime
    of the system -- they define the VSA "alphabet".
    """

    def __init__(self, vsa_dim: int = 10000, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.rng = np.random.RandomState(seed)

        # {category: {name: vector_id}}  -- vector_id is just the name string
        self._primitives: Dict[str, Dict[str, np.ndarray]] = {}

        # Category tag vectors (for compositional binding/unbinding)
        self._tags: Dict[str, np.ndarray] = {}

        self._init_defaults()

    def _rand_bipolar(self) -> np.ndarray:
        return (self.rng.randint(0, 2, self.vsa_dim).astype(np.float32) * 2 - 1)

    def _init_defaults(self):
        """Initialize the fundamental attribute categories."""
        self.add_category("shape", {
            "sphere": None, "cube": None, "cylinder": None,
            "capsule": None, "cone": None, "flat": None,
        })
        self.add_category("color", {
            "red": None, "green": None, "blue": None, "yellow": None,
            "white": None, "black": None, "orange": None, "purple": None,
        })
        self.add_category("mass", {
            "very_light": None, "light": None, "medium": None,
            "heavy": None, "very_heavy": None,
        })
        self.add_category("material", {
            "metal": None, "wood": None, "plastic": None, "rubber": None,
            "glass": None, "stone": None, "fabric": None,
        })
        self.add_category("elasticity", {
            "rigid": None, "stiff": None, "elastic": None,
            "soft": None, "fluid": None,
        })
        self.add_category("texture", {
            "smooth": None, "rough": None, "sticky": None, "slippery": None,
        })
        self.add_category("temperature", {
            "cold": None, "cool": None, "warm": None, "hot": None,
        })
        self.add_category("state", {
            "solid": None, "liquid": None, "gas": None,
        })

    def add_category(self, category: str,
                     primitives: Dict[str, Optional[np.ndarray]]):
        """Add a category with named primitives.

        If a primitive value is None, a random bipolar vector is generated.
        """
        if category not in self._tags:
            self._tags[category] = self._rand_bipolar()

        cat = {}
        for name, vec in primitives.items():
            if vec is None:
                vec = self._rand_bipolar()
            cat[name] = vec
        self._primitives[category] = cat

    def get(self, category: str, name: str) -> Optional[np.ndarray]:
        """Get a primitive vector by category and name."""
        cat = self._primitives.get(category)
        if cat is None:
            return None
        return cat.get(name)

    def get_tag(self, category: str) -> Optional[np.ndarray]:
        """Get the category tag vector."""
        return self._tags.get(category)

    def list_categories(self) -> List[str]:
        return list(self._primitives.keys())

    def list_primitives(self, category: str) -> List[str]:
        cat = self._primitives.get(category, {})
        return list(cat.keys())

    def all_primitive_ids(self) -> List[str]:
        """Return all primitive IDs in 'category_name' format."""
        ids = []
        for cat, prims in self._primitives.items():
            for name in prims:
                ids.append(f"{cat}_{name}")
        return ids

    def resolve_id(self, attr_id: str) -> Tuple[str, str]:
        """Parse 'category_name' into (category, name)."""
        for cat in self._primitives:
            prefix = cat + "_"
            if attr_id.startswith(prefix):
                return cat, attr_id[len(prefix):]
        return "", attr_id

    def get_by_id(self, attr_id: str) -> Optional[np.ndarray]:
        """Get a primitive by its full 'category_name' ID."""
        cat, name = self.resolve_id(attr_id)
        if cat:
            return self.get(cat, name)
        return None

    def __len__(self) -> int:
        return sum(len(v) for v in self._primitives.values())


class RecipeStore:
    """Object recipe store -- the core VSA memory.

    Stores objects as recipes (lists of attribute IDs), NOT as raw vectors.
    Vectors are synthesized on-demand from recipes + primitives.

    Interface:
      1. register_recipe(object_id, [attr_id1, attr_id2, ...])
      2. synthesize(object_id) -> VSA vector (temporary, not stored)
      3. resonate(query_vector) -> best matching object_id
      4. decompose(query_vector) -> {category: (name, confidence)}
    """

    def __init__(self, primitives: AttributePrimitivePool):
        self.primitives = primitives
        self.vsa_dim = primitives.vsa_dim

        # {object_id: [attr_id1, attr_id2, ...]}
        self._recipes: Dict[str, List[str]] = {}

        # Precomputed cache: {object_id: vector} -- lazily invalidated
        self._cache: Dict[str, np.ndarray] = {}

    def register_recipe(self, object_id: str, attr_ids: List[str]):
        """Register or update an object recipe.

        Args:
            object_id: unique identifier (e.g. "ball_123")
            attr_ids:  list of attribute IDs (e.g. ["shape_sphere", "color_red", "material_rubber"])
        """
        self._recipes[object_id] = list(attr_ids)
        self._cache.pop(object_id, None)  # invalidate cache

    def remove_recipe(self, object_id: str):
        self._recipes.pop(object_id, None)
        self._cache.pop(object_id, None)

    def get_recipe(self, object_id: str) -> Optional[List[str]]:
        return self._recipes.get(object_id)

    def list_objects(self) -> List[str]:
        return list(self._recipes.keys())

    def synthesize(self, object_id: str) -> Optional[np.ndarray]:
        """Synthesize a VSA vector from a recipe on-demand.

        The vector is NOT stored permanently -- it's computed from:
          vec = Σ (primitive_i ⊗ tag_i)   for each attr_id in recipe

        Returns None if object_id not found.
        """
        if object_id in self._cache:
            return self._cache[object_id].copy()

        recipe = self._recipes.get(object_id)
        if recipe is None:
            return None

        vec = np.zeros(self.vsa_dim, dtype=np.float32)
        for attr_id in recipe:
            cat, name = self.primitives.resolve_id(attr_id)
            prim = self.primitives.get(cat, name)
            tag = self.primitives.get_tag(cat)
            if prim is not None and tag is not None:
                vec += prim * tag  # bind primitive with category tag

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        self._cache[object_id] = vec
        return vec.copy()

    def synthesize_from_ids(self, attr_ids: List[str]) -> np.ndarray:
        """Synthesize a VSA vector from an ad-hoc list of attribute IDs."""
        vec = np.zeros(self.vsa_dim, dtype=np.float32)
        for attr_id in attr_ids:
            cat, name = self.primitives.resolve_id(attr_id)
            prim = self.primitives.get(cat, name)
            tag = self.primitives.get_tag(cat)
            if prim is not None and tag is not None:
                vec += prim * tag

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def resonate(self, query_vec: np.ndarray, top_k: int = 5
                 ) -> List[Tuple[str, float]]:
        """Resonance search: find the recipes most similar to a query vector.

        Returns: [(object_id, similarity), ...] sorted by similarity descending.
        """
        if not self._recipes:
            return []

        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-10:
            return []

        results = []
        for obj_id in self._recipes:
            vec = self.synthesize(obj_id)
            if vec is None:
                continue
            v_norm = np.linalg.norm(vec)
            if v_norm < 1e-10:
                continue
            sim = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            results.append((obj_id, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def decompose(self, query_vec: np.ndarray
                  ) -> Dict[str, Tuple[str, float]]:
        """Decompose a query vector into attribute categories.

        For each category, unbind the category tag and find the nearest primitive.

        Returns: {category: (best_primitive_name, similarity)}
        """
        results = {}
        q_norm = np.linalg.norm(query_vec)
        if q_norm < 1e-10:
            return results

        for category in self.primitives.list_categories():
            tag = self.primitives.get_tag(category)
            if tag is None:
                continue

            # Unbind: residual = query ⊗ tag (for bipolar, tag^2 = 1)
            residual = query_vec * tag

            best_name = None
            best_sim = -2.0

            for name in self.primitives.list_primitives(category):
                prim = self.primitives.get(category, name)
                if prim is None:
                    continue
                # Compare residual with the raw primitive (not labeled)
                # After unbinding: residual ~ prim + noise
                # The noise from other categories is quasi-orthogonal
                p_norm = np.linalg.norm(prim)
                r_norm = np.linalg.norm(residual)
                if p_norm < 1e-10 or r_norm < 1e-10:
                    continue
                sim = float(np.dot(residual, prim) / (r_norm * p_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_name is not None:
                results[category] = (best_name, best_sim)

        return results

    def find_by_attributes(self, attr_ids: List[str], top_k: int = 5
                           ) -> List[Tuple[str, float]]:
        """Find objects whose recipes contain all specified attribute IDs."""
        target_set = set(attr_ids)
        results = []

        for obj_id, recipe in self._recipes.items():
            overlap = target_set.intersection(set(recipe))
            if overlap:
                # Score = fraction of target attributes matched
                score = len(overlap) / len(target_set)
                results.append((obj_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._recipes)

    def __contains__(self, object_id: str) -> bool:
        return object_id in self._recipes


class ConceptSynthesizer:
    """Higher-level concept synthesis from recipes.

    Supports:
    - Composing ad-hoc concepts (e.g. "red elastic ball") from attribute names
    - Comparing concepts via resonance
    - Explaining concepts by decomposing into attributes
    """

    def __init__(self, store: RecipeStore):
        self.store = store

    def compose_concept(self, attributes: Dict[str, str]) -> np.ndarray:
        """Compose a concept vector from {category: name} dict.

        Example: {"shape": "sphere", "color": "red", "elasticity": "elastic"}
        """
        attr_ids = [f"{cat}_{name}" for cat, name in attributes.items()]
        return self.store.synthesize_from_ids(attr_ids)

    def match_concept(self, query_vec: np.ndarray
                      ) -> Tuple[Optional[str], Dict[str, Tuple[str, float]]]:
        """Match a query vector to the closest stored recipe and decompose it.

        Returns: (best_object_id, decomposition_dict)
        """
        matches = self.store.resonate(query_vec, top_k=1)
        best_id = matches[0][0] if matches else None
        decomposition = self.store.decompose(query_vec)
        return best_id, decomposition

    def explain_concept(self, object_id: str) -> Dict[str, str]:
        """Get human-readable explanation of a recipe.

        Returns: {category: primitive_name}
        """
        recipe = self.store.get_recipe(object_id)
        if recipe is None:
            return {}

        explanation = {}
        for attr_id in recipe:
            cat, name = self.store.primitives.resolve_id(attr_id)
            if cat:
                explanation[cat] = name
        return explanation

    def concept_similarity(self, obj_id_a: str, obj_id_b: str) -> float:
        """Compute cosine similarity between two stored recipes."""
        vec_a = self.store.synthesize(obj_id_a)
        vec_b = self.store.synthesize(obj_id_b)
        if vec_a is None or vec_b is None:
            return 0.0
        na = np.linalg.norm(vec_a)
        nb = np.linalg.norm(vec_b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (na * nb))
