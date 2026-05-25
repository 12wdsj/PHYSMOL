"""PHYSMOL Multimodal Perception Encoders.

Each modality encodes raw sensory data into a VSA vector.
These vectors are then bound together to form multi-modal object concepts.

Supported modalities:
- Vision: image -> CNN/ViT features -> VSA vector
- Audio: collision force + material -> synthesized spectrum -> VSA vector
- Tactile: contact force + position -> VSA vector
- Olfactory: material type -> lookup table -> VSA vector
- Proprioceptive: joint positions/velocities -> VSA vector
"""

import numpy as np
from typing import Dict, Optional, Any

try:
    from . import _vsa
    from .vsa import VectorSymbolicArchitecture, VSAVector
    _HAS_VSA = True
except ImportError:
    _HAS_VSA = False


class ModalityEncoder:
    """Base class for modality-specific encoders."""

    def __init__(self, vsa_dim: int = 4096, modality_name: str = "base"):
        self.vsa_dim = vsa_dim
        self.modality_name = modality_name
        # Random projection matrix: maps raw features to VSA space
        self._projection = None
        self._raw_dim = None

    def _init_projection(self, raw_dim: int, seed: int = 42):
        """Initialize random projection from raw feature space to VSA space."""
        self._raw_dim = raw_dim
        rng = np.random.RandomState(seed)
        # Use sparse random projection (much faster, preserves distances)
        self._projection = rng.choice(
            [-1, 0, 0, 0, 1],  # sparse: 60% zeros
            size=(raw_dim, self.vsa_dim)
        ).astype(np.float32) / np.sqrt(raw_dim)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        """Encode raw features into a VSA vector (numpy array)."""
        raise NotImplementedError

    def encode_raw(self, raw: np.ndarray) -> np.ndarray:
        """Project raw features to VSA space."""
        if self._projection is None:
            self._init_projection(len(raw))
        if len(raw) != self._raw_dim:
            # Pad or truncate
            padded = np.zeros(self._raw_dim, dtype=np.float32)
            padded[:min(len(raw), self._raw_dim)] = raw[:min(len(raw), self._raw_dim)]
            raw = padded
        return raw @ self._projection


class VisionEncoder(ModalityEncoder):
    """Encode visual input (image or color+shape features) into VSA vector.

    Without a trained CNN, uses handcrafted features:
    - Color histogram (HSV bins)
    - Shape descriptors (area, aspect ratio, circularity)
    - Texture (gradient statistics)
    """

    def __init__(self, vsa_dim: int = 4096):
        super().__init__(vsa_dim, "vision")
        self._init_projection(64, seed=100)  # 64-dim handcrafted features

    def encode_color(self, rgb: np.ndarray) -> np.ndarray:
        """Encode RGB color to a feature vector.

        Uses quantized HSV histogram.
        """
        rgb = np.clip(rgb, 0, 1)
        r, g, b = rgb[0], rgb[1], rgb[2]

        # Simple HSV conversion
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        diff = cmax - cmin

        # Hue
        if diff < 1e-6:
            h = 0
        elif cmax == r:
            h = 60 * ((g - b) / diff % 6)
        elif cmax == g:
            h = 60 * ((b - r) / diff + 2)
        else:
            h = 60 * ((r - g) / diff + 4)

        # Saturation
        s = diff / cmax if cmax > 0 else 0

        # Value
        v = cmax

        # Quantize into bins
        h_bin = int(h / 360 * 16) % 16  # 16 hue bins
        s_bin = min(int(s * 4), 3)       # 4 saturation bins
        v_bin = min(int(v * 4), 3)       # 4 value bins

        # One-hot encoding (16+4+4 = 24 dims)
        color_feat = np.zeros(24, dtype=np.float32)
        color_feat[h_bin] = 1.0
        color_feat[16 + s_bin] = 1.0
        color_feat[20 + v_bin] = 1.0

        return color_feat

    def encode_shape(self, shape_type: str, size: float = 1.0,
                     aspect_ratio: float = 1.0) -> np.ndarray:
        """Encode shape descriptor.

        shape_type: "sphere", "box", "cylinder", "capsule", "mesh"
        """
        shapes = {"sphere": 0, "box": 1, "cylinder": 2, "capsule": 3, "mesh": 4}
        shape_onehot = np.zeros(5, dtype=np.float32)
        idx = shapes.get(shape_type, 4)
        shape_onehot[idx] = 1.0

        # Size and aspect ratio
        size_feat = np.array([size, aspect_ratio], dtype=np.float32)

        # Circularity (sphere=1, box=0)
        circularity = np.array([1.0 if shape_type == "sphere" else 0.0], dtype=np.float32)

        return np.concatenate([shape_onehot, size_feat, circularity])

    def encode_from_mujoco(self, model, data, geom_id: int) -> np.ndarray:
        """Encode visual features directly from MuJoCo state.

        Extracts: color (rgba), shape type, size.
        """
        import mujoco

        # Color
        rgba = model.geom_rgba[geom_id]
        color_feat = self.encode_color(rgba[:3])

        # Shape
        geom_type = model.geom_type[geom_id]
        type_map = {0: "plane", 1: "hfield", 2: "sphere", 3: "box",
                    4: "cylinder", 5: "capsule", 6: "ellipsoid", 7: "mesh"}
        shape_type = type_map.get(geom_type, "mesh")
        size = float(np.mean(model.geom_size[geom_id]))
        shape_feat = self.encode_shape(shape_type, size)

        # Texture features (placeholder: use geom ID as proxy)
        texture_feat = np.zeros(32, dtype=np.float32)
        texture_feat[geom_id % 32] = 1.0

        raw = np.concatenate([color_feat, shape_feat, texture_feat])
        return self.encode_raw(raw)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        return self.encode_raw(raw_features)


class AudioEncoder(ModalityEncoder):
    """Encode audio features from collision physics.

    Synthesizes impact sound characteristics from:
    - Collision force magnitude and direction
    - Material properties (hardness, damping)
    - Object size (affects resonant frequency)
    """

    def __init__(self, vsa_dim: int = 4096):
        super().__init__(vsa_dim, "audio")
        self._init_projection(32, seed=200)

    def encode_collision(self, force_magnitude: float, force_direction: np.ndarray,
                         hardness: float = 0.5, size: float = 0.1) -> np.ndarray:
        """Encode a collision event into an audio feature vector.

        Physics: impact sound spectrum is dominated by:
        - Low freq: object resonance ~ c / size (c = speed of sound in material)
        - High freq: impact transient ~ force / mass
        - Decay rate: depends on damping
        """
        # Resonant frequency (rough model)
        c_sound = 3000 * hardness  # m/s, harder = faster sound
        f_resonant = c_sound / max(size, 0.01)  # Hz
        f_resonant = min(f_resonant, 20000)  # cap at 20kHz

        # Impact intensity
        intensity = min(force_magnitude / 100.0, 1.0)  # normalize

        # Spectral centroid (brightness)
        brightness = hardness * 0.7 + intensity * 0.3

        # Decay time
        decay = size / max(hardness, 0.1)  # bigger & softer = longer decay

        # Encode as feature vector
        feat = np.array([
            f_resonant / 20000.0,  # normalized freq
            intensity,
            brightness,
            decay,
            force_direction[0] if len(force_direction) > 0 else 0,
            force_direction[1] if len(force_direction) > 1 else 0,
            force_direction[2] if len(force_direction) > 2 else 0,
            hardness,
            size,
        ], dtype=np.float32)

        # Spectral representation (simplified)
        # 23 frequency bins from 20Hz to 20kHz
        freq_bins = np.logspace(np.log10(20), np.log10(20000), 23)
        spectrum = intensity * np.exp(-0.5 * ((freq_bins - f_resonant) / (f_resonant * 0.3)) ** 2)

        raw = np.concatenate([feat, spectrum.astype(np.float32)])
        return self.encode_raw(raw)

    def encode_from_mujoco(self, data) -> np.ndarray:
        """Encode the most recent collision from MuJoCo contact data."""
        if data.ncon == 0:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        # Find strongest contact
        max_force = 0
        max_contact = None
        for i in range(data.ncon):
            contact = data.contact[i]
            force = np.linalg.norm(data.cfrc_ext[contact.geom1] if contact.geom1 >= 0 else np.zeros(6))
            if force > max_force:
                max_force = force
                max_contact = contact

        if max_contact is None:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        force_dir = data.cfrc_ext[max_contact.geom1][:3] if max_contact.geom1 >= 0 else np.zeros(3)
        force_mag = np.linalg.norm(force_dir)
        if force_mag > 0:
            force_dir = force_dir / force_mag

        return self.encode_collision(force_mag, force_dir, hardness=0.5, size=0.1)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        return self.encode_raw(raw_features)


class TactileEncoder(ModalityEncoder):
    """Encode tactile features from contact forces.

    Tactile encoding captures:
    - Contact force magnitude and direction
    - Contact area
    - Pressure distribution
    - Slip detection (tangential vs normal force ratio)
    """

    def __init__(self, vsa_dim: int = 4096):
        super().__init__(vsa_dim, "tactile")
        self._init_projection(16, seed=300)

    def encode_contact(self, force: np.ndarray, position: np.ndarray,
                       normal: np.ndarray, area: float = 0.01) -> np.ndarray:
        """Encode a single contact point."""
        force_mag = np.linalg.norm(force)

        # Decompose force into normal and tangential
        if np.linalg.norm(normal) > 0:
            normal_norm = normal / np.linalg.norm(normal)
            f_normal = np.dot(force, normal_norm)
            f_tangential = force - f_normal * normal_norm
            slip_ratio = np.linalg.norm(f_tangential) / max(abs(f_normal), 1e-6)
        else:
            f_normal = force_mag
            slip_ratio = 0

        raw = np.array([
            force_mag,
            f_normal,
            slip_ratio,
            area,
            position[0], position[1], position[2],
            normal[0] if len(normal) > 0 else 0,
            normal[1] if len(normal) > 1 else 0,
            normal[2] if len(normal) > 2 else 0,
            0, 0, 0, 0, 0, 0,  # padding to 16
        ], dtype=np.float32)[:16]

        return self.encode_raw(raw)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        return self.encode_raw(raw_features)


class OlfactoryEncoder(ModalityEncoder):
    """Encode olfactory (smell) features.

    Since there's no physics-based smell simulator, we use a lookup table
    mapping material/chemical properties to odor vectors.

    This is a simplified model: real olfaction involves thousands of
    receptor types. Here we use a material-based encoding.
    """

    # Material odor profiles (simplified)
    ODOR_PROFILES = {
        "metal":     np.array([0.1, 0.8, 0.0, 0.0, 0.3, 0.0, 0.0, 0.5]),
        "wood":      np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.3, 0.0, 0.0]),
        "plastic":   np.array([0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0]),
        "rubber":    np.array([0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.5, 0.0]),
        "fabric":    np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0]),
        "organic":   np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.2]),
        "water":     np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
        "stone":     np.array([0.2, 0.3, 0.0, 0.0, 0.5, 0.0, 0.0, 0.4]),
        "glass":     np.array([0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
        "food":      np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "none":      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }

    # Odor dimensions:
    # [pungent, metallic, woody, chemical, earthy, floral, smoky, mineral]

    def __init__(self, vsa_dim: int = 4096):
        super().__init__(vsa_dim, "olfactory")
        self._init_projection(8, seed=400)

    def encode_material(self, material: str) -> np.ndarray:
        """Encode a material's odor profile."""
        profile = self.ODOR_PROFILES.get(material, self.ODOR_PROFILES["none"])
        # Add noise to simulate variability
        noise = np.random.randn(8).astype(np.float32) * 0.05
        raw = np.clip(profile + noise, 0, 1)
        return self.encode_raw(raw)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        return self.encode_raw(raw_features)


class ProprioceptiveEncoder(ModalityEncoder):
    """Encode proprioceptive state (joint positions, velocities)."""

    def __init__(self, vsa_dim: int = 4096, max_joints: int = 20):
        super().__init__(vsa_dim, "proprioceptive")
        self.max_joints = max_joints
        self._init_projection(max_joints * 2, seed=500)  # pos + vel per joint

    def encode_state(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Encode joint positions and velocities."""
        # Pad to max_joints
        pos = np.zeros(self.max_joints, dtype=np.float32)
        vel = np.zeros(self.max_joints, dtype=np.float32)
        n = min(len(qpos), self.max_joints)
        pos[:n] = qpos[:n]
        vel[:n] = qvel[:n]

        raw = np.concatenate([pos, vel])
        return self.encode_raw(raw)

    def encode(self, raw_features: np.ndarray) -> np.ndarray:
        return self.encode_raw(raw_features)


class MultiModalPerception:
    """Unified multimodal perception system.

    Combines all modality encoders and produces composite VSA vectors
    for objects and scenes.
    """

    def __init__(self, vsa_dim: int = 4096):
        self.vsa_dim = vsa_dim
        self.vision = VisionEncoder(vsa_dim)
        self.audio = AudioEncoder(vsa_dim)
        self.tactile = TactileEncoder(vsa_dim)
        self.olfactory = OlfactoryEncoder(vsa_dim)
        self.proprioceptive = ProprioceptiveEncoder(vsa_dim)

    def encode_object(self, visual: Optional[np.ndarray] = None,
                      audio: Optional[np.ndarray] = None,
                      tactile: Optional[np.ndarray] = None,
                      olfactory: Optional[str] = None,
                      proprioceptive: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode an object by binding available modality vectors.

        Uses bundling for multiple modalities: the more modalities agree,
        the stronger the concept.
        """
        vectors = []

        if visual is not None:
            v = self.vision.encode_raw(visual) if visual.ndim == 1 else self.vision.encode(visual)
            vectors.append(v)

        if audio is not None:
            v = self.audio.encode_raw(audio) if audio.ndim == 1 else self.audio.encode(audio)
            vectors.append(v)

        if tactile is not None:
            v = self.tactile.encode_raw(tactile) if tactile.ndim == 1 else self.tactile.encode(tactile)
            vectors.append(v)

        if olfactory is not None:
            v = self.olfactory.encode_material(olfactory)
            vectors.append(v)

        if proprioceptive is not None:
            v = self.proprioceptive.encode_raw(proprioceptive) if proprioceptive.ndim == 1 else self.proprioceptive.encode(proprioceptive)
            vectors.append(v)

        if not vectors:
            return np.zeros(self.vsa_dim, dtype=np.float32)

        # Bundle all modality vectors
        result = vectors[0].copy()
        for v in vectors[1:]:
            result = result + v

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm

        return result

    def encode_from_mujoco(self, model, data, body_id: int) -> np.ndarray:
        """Encode a MuJoCo object using all available sensory channels."""
        import mujoco

        # Vision: from geom properties
        geom_id = -1
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == body_id:
                geom_id = i
                break

        visual = None
        if geom_id >= 0:
            try:
                visual = self.vision.encode_from_mujoco(model, data, geom_id)
            except Exception:
                pass

        # Audio: from contacts
        audio = None
        try:
            audio = self.audio.encode_from_mujoco(data)
        except Exception:
            pass

        # Tactile: from contact forces on this body
        tactile = None
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                force = data.cfrc_ext[geom_id][:3] if geom_id >= 0 else np.zeros(3)
                pos = contact.pos
                normal = contact.frame[:3]
                tactile = self.tactile.encode_contact(force, pos, normal)
                break

        # Proprioceptive: body position and velocity
        proprio = self.proprioceptive.encode_state(
            data.xpos[body_id], data.cvel[body_id][:3]
        )

        return self.encode_object(
            visual=visual, audio=audio, tactile=tactile,
            olfactory=None, proprioceptive=proprio
        )

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two composite vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(dot / (norm_a * norm_b))
