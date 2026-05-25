"""PHYSMOL VSA Concept System: Primitives, Composition, and Fields.

Core design (from PHYSMOL paper):
- Codebook stores ATOMIC PRIMITIVES (color, shape, mass, material...)
- Objects are COMPOSED by binding primitives together
- Identification = unbind + nearest-neighbor search in codebook
- Fields (force, light, magnetic) = spatial sampling with FPE position encoding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .perception import MultiModalPerception


class PrimitiveCodebook:
    """Codebook of atomic VSA primitives organized by attribute category.

    Each category (color, shape, mass, material...) has its own set of
    randomly generated base vectors. Objects are composed by binding
    one primitive from each relevant category.
    """

    def __init__(self, vsa_dim: int = 4096, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Category -> {name: vector}
        self.categories: Dict[str, Dict[str, np.ndarray]] = {}

        # Category label vectors (used to make primitives recoverable)
        # Each category gets a random "tag" vector. Primitives are stored
        # as (primitive_vec ⊗ category_tag), then bundled.
        # To decompose: unbind by category_tag, then search codebook.
        self.category_tags: Dict[str, np.ndarray] = {}

        # Initialize default categories
        self._init_default_categories()

    def _random_bipolar(self) -> np.ndarray:
        """Generate a random bipolar vector (-1/+1)."""
        return (self.rng.randint(0, 2, self.vsa_dim).astype(np.float32) * 2 - 1)

    def _init_default_categories(self):
        """Initialize the fundamental attribute categories."""

        # Color primitives
        self.category_tags["color"] = self._random_bipolar()
        self.add_category("color", {
            "red": self._random_bipolar(),
            "green": self._random_bipolar(),
            "blue": self._random_bipolar(),
            "yellow": self._random_bipolar(),
            "white": self._random_bipolar(),
            "black": self._random_bipolar(),
            "orange": self._random_bipolar(),
            "purple": self._random_bipolar(),
        })

        # Shape primitives
        self.category_tags["shape"] = self._random_bipolar()
        self.add_category("shape", {
            "sphere": self._random_bipolar(),
            "cube": self._random_bipolar(),
            "cylinder": self._random_bipolar(),
            "capsule": self._random_bipolar(),
            "cone": self._random_bipolar(),
            "flat": self._random_bipolar(),    # plane/plate
            "elongated": self._random_bipolar(),  # rod/stick
            "irregular": self._random_bipolar(),
        })

        # Mass class primitives (discretized)
        self.category_tags["mass"] = self._random_bipolar()
        self.add_category("mass", {
            "very_light": self._random_bipolar(),   # < 0.05 kg
            "light": self._random_bipolar(),         # 0.05-0.2 kg
            "medium": self._random_bipolar(),        # 0.2-1.0 kg
            "heavy": self._random_bipolar(),         # 1.0-5.0 kg
            "very_heavy": self._random_bipolar(),    # > 5.0 kg
        })

        # Elasticity primitives
        self.category_tags["elasticity"] = self._random_bipolar()
        self.add_category("elasticity", {
            "rigid": self._random_bipolar(),       # steel, stone
            "stiff": self._random_bipolar(),       # wood, hard plastic
            "elastic": self._random_bipolar(),     # rubber, spring
            "soft": self._random_bipolar(),        # foam, cloth
            "fluid": self._random_bipolar(),       # water, sand
        })

        # Material primitives
        self.category_tags["material"] = self._random_bipolar()
        self.add_category("material", {
            "metal": self._random_bipolar(),
            "wood": self._random_bipolar(),
            "plastic": self._random_bipolar(),
            "rubber": self._random_bipolar(),
            "glass": self._random_bipolar(),
            "stone": self._random_bipolar(),
            "fabric": self._random_bipolar(),
            "organic": self._random_bipolar(),
        })

        # Surface texture primitives
        self.category_tags["texture"] = self._random_bipolar()
        self.add_category("texture", {
            "smooth": self._random_bipolar(),
            "rough": self._random_bipolar(),
            "sticky": self._random_bipolar(),
            "slippery": self._random_bipolar(),
            "gritty": self._random_bipolar(),
        })

        # Temperature primitives
        self.category_tags["temperature"] = self._random_bipolar()
        self.add_category("temperature", {
            "cold": self._random_bipolar(),
            "cool": self._random_bipolar(),
            "warm": self._random_bipolar(),
            "hot": self._random_bipolar(),
        })

    def add_category(self, name: str, primitives: Dict[str, np.ndarray]):
        """Add or update a category of primitives."""
        self.categories[name] = primitives

    def get_primitive(self, category: str, name: str) -> Optional[np.ndarray]:
        """Get a specific primitive vector."""
        cat = self.categories.get(category)
        if cat is None:
            return None
        return cat.get(name)

    def compose(self, properties: Dict[str, str]) -> np.ndarray:
        """Compose an object concept by BUNDLING tagged primitives.

        Design: object = Σ (primitive_i ⊗ category_tag_i)

        This makes decomposition possible:
        - To recover "color": unbind (object ⊗ color_tag), then search color codebook
        - Tags for different categories are quasi-orthogonal, so cross-talk is minimal

        Args:
            properties: {category: primitive_name}
                e.g., {"color": "red", "shape": "sphere", "mass": "heavy"}

        Returns:
            Composed VSA vector (bundling of tagged primitives)
        """
        result = np.zeros(self.vsa_dim, dtype=np.float32)
        for category, name in properties.items():
            prim = self.get_primitive(category, name)
            if prim is None:
                continue
            tag = self.category_tags.get(category)
            if tag is None:
                tag = self._random_bipolar()
                self.category_tags[category] = tag
            # Labeled primitive: primitive ⊗ category_tag
            labeled = prim * tag
            result = result + labeled  # bundle (superposition)

        return result

    def decompose(self, object_vec: np.ndarray,
                  categories: Optional[List[str]] = None
                  ) -> Dict[str, Tuple[str, float]]:
        """Decompose an object vector by UNBINDING category tags, then searching.

        Process for each category:
        1. Unbind: residual = object_vec ⊗ category_tag
           (removes this category's contribution, isolates the primitive)
        2. Search: find the primitive in this category most similar to residual

        Args:
            object_vec: The composed object VSA vector
            categories: Which categories to decompose (default: all)

        Returns:
            {category: (best_primitive_name, similarity)}
        """
        if categories is None:
            categories = list(self.categories.keys())

        result = {}
        for category in categories:
            cat = self.categories.get(category, {})
            if not cat:
                continue

            tag = self.category_tags.get(category)
            if tag is None:
                continue

            # Unbind: for bipolar, inverse(tag) = tag, so unbind = bind
            residual = object_vec * tag

            # Find nearest primitive in this category
            best_name = None
            best_sim = -2.0

            for name, prim in cat.items():
                # The labeled primitive was: prim ⊗ tag
                # After unbinding with tag: (prim ⊗ tag) ⊗ tag = prim ⊗ (tag ⊗ tag) = prim
                # (since bipolar: tag² = 1)
                labeled_prim = prim * tag
                sim = float(np.dot(residual, labeled_prim) / (
                    np.linalg.norm(residual) * np.linalg.norm(labeled_prim) + 1e-10))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            result[category] = (best_name, best_sim)

        return result

    def nearest_in_category(self, query: np.ndarray, category: str,
                            top_k: int = 3) -> List[Tuple[str, float]]:
        """Find the k nearest primitives in a category."""
        cat = self.categories.get(category, {})
        if not cat:
            return []

        sims = []
        for name, prim in cat.items():
            sim = float(np.dot(query, prim) / (
                np.linalg.norm(query) * np.linalg.norm(prim) + 1e-10))
            sims.append((name, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def __repr__(self):
        total = sum(len(v) for v in self.categories.values())
        return f"PrimitiveCodebook({len(self.categories)} categories, {total} primitives)"


class FieldEncoder:
    """Encode continuous spatial fields (force, light, magnetic, etc.) using VSA.

    A field is represented as a set of (position, value) samples:
        Field = Σ_i [ H(x_i, y_i, z_i) ⊗ V(field_value_i) ]

    where H is the FPE-encoded position and V is the field value primitive.

    This allows:
    - Querying "what's the field value at position (x,y,z)?"
    - Comparing two fields
    - Composing fields (superposition)
    """

    def __init__(self, vsa_dim: int = 4096, seed: int = 42):
        self.vsa_dim = vsa_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # FPE base vectors for spatial encoding
        self.base_x = self._random_phase()
        self.base_y = self._random_phase()
        self.base_z = self._random_phase()

        # Field value primitives (intensity levels)
        self.value_primitives = {}
        self._init_value_primitives()

        # Field type primitives
        self.field_types = {
            "gravity": self._random_bipolar(),
            "electromagnetic": self._random_bipolar(),
            "light": self._random_bipolar(),
            "thermal": self._random_bipolar(),
            "acoustic": self._random_bipolar(),
            "pressure": self._random_bipolar(),
            "chemical": self._random_bipolar(),  # smell diffusion
        }

    def _random_phase(self) -> np.ndarray:
        return self.rng.uniform(0, 2 * np.pi, self.vsa_dim).astype(np.float32)

    def _random_bipolar(self) -> np.ndarray:
        return (self.rng.randint(0, 2, self.vsa_dim).astype(np.float32) * 2 - 1)

    def _init_value_primitives(self):
        """Initialize field value primitives for different intensity levels."""
        levels = ["zero", "very_weak", "weak", "moderate", "strong", "very_strong", "extreme"]
        for level in levels:
            self.value_primitives[level] = self._random_bipolar()

    def encode_position(self, x: float, y: float, z: float) -> np.ndarray:
        """FPE encode a 3D position: H(s) = B_x^x * B_y^y * B_z^z

        In FHRR (Fourier Holographic Reduced Representation):
        Each element is a complex phase, binding is element-wise multiplication.
        For real-valued approximation, we store combined phase angles.
        """
        return x * self.base_x + y * self.base_y + z * self.base_z

    def encode_field_value(self, value: float) -> np.ndarray:
        """Encode a scalar field value as a VSA vector.

        Uses graded encoding: value selects and blends nearby primitives.
        """
        # Map value to two nearest levels and interpolate
        levels = list(self.value_primitives.keys())
        n = len(levels)

        # Normalize value to [0, 1] range (assuming max ~10 for force, ~1 for normalized)
        v = max(0.0, min(1.0, value / 10.0))

        idx_f = v * (n - 1)
        idx_low = int(idx_f)
        idx_high = min(idx_low + 1, n - 1)
        alpha = idx_f - idx_low

        vec_low = self.value_primitives[levels[idx_low]]
        vec_high = self.value_primitives[levels[idx_high]]

        # Bundled interpolation
        return (1 - alpha) * vec_low + alpha * vec_high

    def encode_field_type(self, field_type: str) -> np.ndarray:
        """Get the primitive for a field type (gravity, EM, light, etc.)."""
        return self.field_types.get(field_type, np.zeros(self.vsa_dim, dtype=np.float32))

    def sample_field(self, field_type: str, x: float, y: float, z: float,
                     value: float) -> np.ndarray:
        """Encode one sample point of a field: type ⊗ position ⊗ value."""
        type_vec = self.encode_field_type(field_type)
        pos_vec = self.encode_position(x, y, z)
        val_vec = self.encode_field_value(value)

        # Triple binding: type ⊗ position ⊗ value
        return type_vec * pos_vec * val_vec

    def encode_field(self, field_type: str,
                     samples: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Encode an entire field from sampled points.

        Args:
            field_type: "gravity", "light", "electromagnetic", etc.
            samples: list of (x, y, z, value) tuples

        Returns:
            Composite field VSA vector (bundling of all samples)
        """
        if not samples:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        result = np.zeros(self.vsa_dim, dtype=np.float32)
        for x, y, z, value in samples:
            result += self.sample_field(field_type, x, y, z, value)

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result

    def query_position(self, field_vec: np.ndarray, x: float, y: float, z: float
                       ) -> float:
        """Estimate field value at a position by similarity search.

        This is approximate: it checks how much the position's encoding
        is present in the field vector.
        """
        pos_vec = self.encode_position(x, y, z)

        # Try each value level and find best match
        best_sim = -2.0
        best_value = 0.0

        for level_name, level_vec in self.value_primitives.items():
            # Construct what this (position, value) would look like
            test_vec = pos_vec * level_vec
            sim = float(np.dot(field_vec, test_vec) / (
                np.linalg.norm(field_vec) * np.linalg.norm(test_vec) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                # Map level name back to value
                level_map = {"zero": 0, "very_weak": 0.15, "weak": 0.3,
                             "moderate": 0.5, "strong": 0.7, "very_strong": 0.85, "extreme": 1.0}
                best_value = level_map.get(level_name, 0.5) * 10.0

        return best_value

    def compare_fields(self, field_a: np.ndarray, field_b: np.ndarray) -> float:
        """Cosine similarity between two field representations."""
        return float(np.dot(field_a, field_b) / (
            np.linalg.norm(field_a) * np.linalg.norm(field_b) + 1e-10))


class ObjectConcept:
    """A fully composed object concept with primitives + fields.

    Structure:
        concept = primitives ⊗ spatial_signature ⊗ field_contributions

    where:
        primitives = color ⊗ shape ⊗ mass ⊗ material ⊗ ...
        spatial_signature = FPE(position) (where the object is)
        field_contributions = object's own field emissions (heat, smell, etc.)
    """

    def __init__(self, name: str, vsa_dim: int = 4096):
        self.name = name
        self.vsa_dim = vsa_dim
        self.primitive_vec = np.zeros(vsa_dim, dtype=np.float32)
        self.position_vec = np.zeros(vsa_dim, dtype=np.float32)
        self.field_vecs = {}  # field_type -> vector
        self.properties = {}  # category -> primitive_name (for human-readable output)

    def set_primitives(self, codebook: PrimitiveCodebook,
                       properties: Dict[str, str]):
        """Set object primitives from property dict."""
        self.properties = properties
        self.primitive_vec = codebook.compose(properties)

    def set_position(self, field_encoder: FieldEncoder,
                     x: float, y: float, z: float):
        """Set the object's spatial position."""
        self.position_vec = field_encoder.encode_position(x, y, z)

    def add_field_emission(self, field_encoder: FieldEncoder,
                           field_type: str,
                           samples: List[Tuple[float, float, float, float]]):
        """Add a field that this object emits (e.g., heat, smell)."""
        self.field_vecs[field_type] = field_encoder.encode_field(field_type, samples)

    def get_full_concept(self) -> np.ndarray:
        """Get the complete concept vector: primitives ⊗ position ⊗ fields."""
        result = self.primitive_vec * self.position_vec

        for field_vec in self.field_vecs.values():
            result = result + field_vec  # bundle field contributions

        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result

    def __repr__(self):
        props = ", ".join(f"{k}={v}" for k, v in self.properties.items())
        return f"ObjectConcept({self.name}: {props})"
