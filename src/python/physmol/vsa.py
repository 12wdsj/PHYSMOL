"""PHYSMOL Vector Symbolic Architecture - Python wrapper."""

import numpy as np
from typing import Optional, Dict, Tuple

# Import the C extension (built via pybind11)
try:
    from . import _vsa
except ImportError:
    raise ImportError(
        "C extension _vsa not found. Build with: python setup.py build_ext --inplace"
    )


class VSAVector:
    """A vector in the VSA hyperspace. Wraps a numpy float32 array."""

    __slots__ = ("_data",)

    def __init__(self, data: np.ndarray):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim != 1:
            raise ValueError("VSAVector data must be 1D")
        self._data = data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def dim(self) -> int:
        return len(self._data)

    def __repr__(self):
        return f"VSAVector(dim={self.dim}, norm={np.linalg.norm(self._data):.4f})"

    def __eq__(self, other):
        if not isinstance(other, VSAVector):
            return NotImplemented
        return np.array_equal(self._data, other._data)


class VectorSymbolicArchitecture:
    """Main VSA interface. Uses C core for all heavy computation."""

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def random_bipolar(self, seed: int = 0) -> VSAVector:
        """Create a random bipolar vector (-1/+1)."""
        return VSAVector(_vsa.random_bipolar(self.dim, seed))

    def random_phase(self, seed: int = 0) -> VSAVector:
        """Create a random phase vector (for FHRR)."""
        return VSAVector(_vsa.random_phase(self.dim, seed))

    def bind(self, a: VSAVector, b: VSAVector) -> VSAVector:
        """Binding (Hadamard product): compose two features."""
        return VSAVector(_vsa.bind(a.data, b.data))

    def bundle(self, a: VSAVector, b: VSAVector) -> VSAVector:
        """Bundling (addition): superpose two concepts."""
        return VSAVector(_vsa.bundle(a.data, b.data))

    def unbind(self, a: VSAVector, b: VSAVector) -> VSAVector:
        """Unbinding: decompose a bound vector."""
        return VSAVector(_vsa.unbind(a.data, b.data))

    def similarity(self, a: VSAVector, b: VSAVector) -> float:
        """Cosine similarity between two vectors."""
        return _vsa.cosine_similarity(a.data, b.data)

    def hamming_distance(self, a: VSAVector, b: VSAVector) -> float:
        """Normalized Hamming distance for bipolar vectors."""
        return _vsa.hamming_distance(a.data, b.data)

    def permute(self, vec: VSAVector, shift: int) -> VSAVector:
        """Circular permutation (positional encoding)."""
        return VSAVector(_vsa.permute(vec.data, shift))

    def fpe_encode(self, x: float, y: float, z: float,
                   base_x: VSAVector, base_y: VSAVector, base_z: VSAVector) -> VSAVector:
        """FPE encode a 3D coordinate into a hypervector."""
        return VSAVector(_vsa.fpe_encode(x, y, z, base_x.data, base_y.data, base_z.data))

    def normalize(self, vec: VSAVector) -> VSAVector:
        """Return a normalized copy."""
        data = vec.data.copy()
        _vsa.normalize(data)
        return VSAVector(data)

    def quantize(self, vec: VSAVector, bits: int = 8) -> VSAVector:
        """Return a quantized copy."""
        data = vec.data.copy()
        _vsa.quantize(data, bits)
        return VSAVector(data)


class Codebook:
    """Named collection of VSA primitive vectors."""

    def __init__(self, dim: int = 4096, capacity: int = 256):
        self._cb = _vsa.Codebook(dim, capacity)
        self.dim = dim

    def add(self, name: str, vec: VSAVector) -> int:
        """Add a named primitive. Returns index."""
        return self._cb.add(name, vec.data)

    def lookup(self, name: str) -> Optional[VSAVector]:
        """Look up a primitive by name."""
        data = self._cb.lookup(name)
        if data is None:
            return None
        return VSAVector(data)

    def nearest(self, query: VSAVector) -> Tuple[int, float, str]:
        """Find most similar primitive. Returns (index, similarity, name)."""
        return self._cb.nearest(query.data)

    def __len__(self) -> int:
        return len(self._cb)

    def __contains__(self, name: str) -> bool:
        return self.lookup(name) is not None

    def __repr__(self):
        return f"Codebook(dim={self.dim}, size={len(self)})"


class FHRRSpace:
    """Fourier Holographic Reduced Representation space for FPE encoding."""

    def __init__(self, dim: int = 4096, seed: int = 42):
        self.dim = dim
        self.vsa = VectorSymbolicArchitecture(dim)
        self.rng = np.random.RandomState(seed)
        # Base vectors for each axis
        self.base_x = self.vsa.random_phase(seed=seed)
        self.base_y = self.vsa.random_phase(seed=seed + 1)
        self.base_z = self.vsa.random_phase(seed=seed + 2)

    def encode_position(self, x: float, y: float, z: float) -> VSAVector:
        """Encode a 3D position into a hypervector via FPE."""
        return self.vsa.fpe_encode(x, y, z, self.base_x, self.base_y, self.base_z)

    def encode_shift(self, dx: float, dy: float, dz: float) -> VSAVector:
        """Encode a spatial displacement."""
        return self.encode_position(dx, dy, dz)

    def translate(self, pos_vec: VSAVector, dx: float, dy: float, dz: float) -> VSAVector:
        """Translate a position vector by (dx, dy, dz) via binding."""
        shift = self.encode_shift(dx, dy, dz)
        return self.vsa.bind(pos_vec, shift)
