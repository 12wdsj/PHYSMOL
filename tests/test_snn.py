"""PHYSMOL SNN Python Integration Tests."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

from physmol.snn import SpikingNetwork, SpikeTrain, CausalGraph, ThreeFactorLearner


class TestSpikeTrain:
    def test_create(self):
        st = SpikeTrain(512)
        assert st.num_neurons == 512
        assert st.count() == 0

    def test_from_array(self):
        st = SpikeTrain(64)
        arr = np.zeros(64, dtype=np.int32)
        arr[0] = 1; arr[10] = 1; arr[50] = 1
        st.from_array(arr)
        assert st.count() == 3

    def test_to_array(self):
        st = SpikeTrain(64)
        arr = np.zeros(64, dtype=np.int32)
        arr[5] = 1
        st.from_array(arr)
        out = st.to_array()
        assert out[5] == 1
        assert out[0] == 0


class TestSpikingNetwork:
    def test_create(self):
        net = SpikingNetwork(256, 256)
        assert net.num_pre == 256
        assert net.num_post == 256

    def test_step(self):
        net = SpikingNetwork(64, 64)
        input_current = np.ones(64, dtype=np.float32) * 0.5
        pre, post = net.step(input_current)
        assert len(pre) == 64
        assert len(post) == 64

    def test_stdp(self):
        net = SpikingNetwork(32, 32)
        w_before = net.weight_matrix.copy()
        input_current = np.ones(32, dtype=np.float32) * 2.0
        net.step(input_current)
        net.stdp()
        w_after = net.weight_matrix
        # Weights should change
        assert not np.allclose(w_before, w_after)


class TestCausalGraph:
    def test_create(self):
        g = CausalGraph(100)
        assert g.edge_count == 0

    def test_add_edge(self):
        g = CausalGraph(100)
        g.add_edge(0, 1, weight=1.0, credit=0.5)
        g.add_edge(1, 2, weight=0.8, credit=0.3)
        assert g.edge_count == 2

    def test_prune(self):
        g = CausalGraph(100)
        g.add_edge(0, 1, weight=1.0, credit=0.5)
        g.add_edge(1, 2, weight=0.8, credit=0.1)
        removed = g.prune(0.3)
        assert removed == 1
        assert g.edge_count == 1

    def test_propagate(self):
        g = CausalGraph(100)
        g.add_edge(0, 1, weight=1.0, credit=0.5)
        g.add_edge(1, 2, weight=0.5, credit=0.5)
        result = g.propagate(0, steps=2)
        assert result[0] > 0  # source activated
        assert result[1] > 0  # first hop
        assert result[2] > 0  # second hop


class TestThreeFactorLearner:
    def test_create(self):
        learner = ThreeFactorLearner(32, 32)
        assert learner.num_pre == 32
        assert learner.num_post == 32

    def test_update(self):
        learner = ThreeFactorLearner(16, 16)
        w_before = learner.weight_matrix.copy()

        pre = SpikeTrain(16)
        post = SpikeTrain(16)
        pre.from_array(np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32))
        post.from_array(np.array([0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.int32))

        learner.update(pre, post, reward=1.0, eta=0.01)
        w_after = learner.weight_matrix
        assert not np.allclose(w_before, w_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
