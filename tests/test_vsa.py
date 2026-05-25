"""PHYSMOL VSA Python Integration Tests."""

import numpy as np
import pytest
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from physmol.vsa import VectorSymbolicArchitecture, VSAVector, Codebook, FHRRSpace


class TestVSAVector:
    def test_create(self):
        vsa = VectorSymbolicArchitecture(4096)
        v = vsa.random_bipolar()
        assert v.dim == 4096
        assert len(v.data) == 4096
        assert v.data.dtype == np.float32

    def test_bind(self):
        vsa = VectorSymbolicArchitecture(4096)
        a = vsa.random_bipolar(1)
        b = vsa.random_bipolar(2)
        c = vsa.bind(a, b)
        # Verify Hadamard product
        expected = a.data * b.data
        np.testing.assert_allclose(c.data, expected, atol=1e-6)

    def test_bind_commutativity(self):
        vsa = VectorSymbolicArchitecture(4096)
        a = vsa.random_bipolar(10)
        b = vsa.random_bipolar(20)
        ab = vsa.bind(a, b)
        ba = vsa.bind(b, a)
        np.testing.assert_allclose(ab.data, ba.data, atol=1e-6)

    def test_bundle(self):
        vsa = VectorSymbolicArchitecture(4096)
        a = vsa.random_bipolar(3)
        b = vsa.random_bipolar(4)
        c = vsa.bundle(a, b)
        expected = a.data + b.data
        np.testing.assert_allclose(c.data, expected, atol=1e-6)

    def test_similarity_self(self):
        vsa = VectorSymbolicArchitecture(4096)
        a = vsa.random_bipolar(5)
        sim = vsa.similarity(a, a)
        assert abs(sim - 1.0) < 1e-4

    def test_similarity_random(self):
        vsa = VectorSymbolicArchitecture(4096)
        a = vsa.random_bipolar(100)
        b = vsa.random_bipolar(200)
        sim = vsa.similarity(a, b)
        assert abs(sim) < 0.1  # nearly orthogonal

    def test_permute(self):
        vsa = VectorSymbolicArchitecture(4096)
        v = vsa.random_bipolar(7)
        shifted = vsa.permute(v, 1)
        assert shifted.data[0] == v.data[1]
        assert shifted.data[-1] == v.data[0]

    def test_normalize(self):
        vsa = VectorSymbolicArchitecture(4096)
        v = vsa.random_bipolar(8)
        normed = vsa.normalize(v)
        norm = np.linalg.norm(normed.data)
        assert abs(norm - 1.0) < 1e-4

    def test_quantize(self):
        vsa = VectorSymbolicArchitecture(4096)
        v = vsa.random_bipolar(9)
        q = vsa.quantize(v, bits=8)
        # Should be close to original
        diff = np.max(np.abs(v.data - q.data))
        assert diff < 0.1


class TestCodebook:
    def test_add_lookup(self):
        cb = Codebook(4096)
        vsa = VectorSymbolicArchitecture(4096)
        red = vsa.random_bipolar(100)
        blue = vsa.random_bipolar(200)

        cb.add("red", red)
        cb.add("blue", blue)
        assert len(cb) == 2

        found = cb.lookup("red")
        assert found is not None
        np.testing.assert_allclose(found.data, red.data, atol=1e-6)

    def test_nearest(self):
        cb = Codebook(4096)
        vsa = VectorSymbolicArchitecture(4096)
        red = vsa.random_bipolar(100)
        blue = vsa.random_bipolar(200)
        cb.add("red", red)
        cb.add("blue", blue)

        idx, sim, name = cb.nearest(red)
        assert name == "red"
        assert sim > 0.99


class TestFHRRSpace:
    def test_encode_position(self):
        fhrr = FHRRSpace(4096)
        v1 = fhrr.encode_position(1.0, 2.0, 3.0)
        v2 = fhrr.encode_position(1.0, 2.0, 3.0)
        # Same position should give same vector
        np.testing.assert_allclose(v1.data, v2.data, atol=1e-6)

    def test_different_positions(self):
        fhrr = FHRRSpace(4096)
        v1 = fhrr.encode_position(0, 0, 0)
        v2 = fhrr.encode_position(1, 0, 0)
        vsa = VectorSymbolicArchitecture(4096)
        sim = vsa.similarity(v1, v2)
        # Different positions should be different
        assert sim < 0.99

    def test_translate(self):
        fhrr = FHRRSpace(4096)
        v1 = fhrr.encode_position(0, 0, 0)
        v2 = fhrr.translate(v1, 1, 0, 0)
        v3 = fhrr.encode_position(1, 0, 0)
        vsa = VectorSymbolicArchitecture(4096)
        # Translation should produce similar result to direct encoding
        sim = vsa.similarity(v2, v3)
        # They won't be identical due to binding noise, but should be correlated
        assert sim > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
