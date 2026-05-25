"""PHYSMOL Spiking Neural Network - Python wrapper."""

import numpy as np
from typing import Optional, List, Tuple

try:
    from . import _snn
except ImportError:
    raise ImportError(
        "C extension _snn not found. Build with: python setup.py build_ext --inplace"
    )


class LIFNeuronParams:
    """Leaky Integrate-and-Fire neuron parameters."""

    def __init__(self, v_threshold=1.0, v_reset=0.0,
                 tau_mem=20.0, tau_ref=2.0, dt=1.0):
        self._params = _snn.LIFParams(v_threshold, v_reset, tau_mem, tau_ref, dt)

    @property
    def v_threshold(self): return self._params.v_threshold
    @property
    def tau_mem(self): return self._params.tau_mem
    @property
    def dt(self): return self._params.dt


class SpikeTrain:
    """Bit-compressed spike state for a population of neurons."""

    def __init__(self, num_neurons: int):
        self._state = _snn.SpikeState(num_neurons)
        self.num_neurons = num_neurons

    def clear(self):
        self._state.clear()

    def count(self) -> int:
        return self._state.count()

    def to_array(self) -> np.ndarray:
        """Return spikes as int32 array (0/1)."""
        return self._state.to_array().astype(np.int32)

    def from_array(self, arr: np.ndarray):
        """Set spikes from int array."""
        self._state.from_array(arr.astype(np.int32))

    @property
    def spikes(self) -> np.ndarray:
        return self.to_array()

    def __repr__(self):
        return f"SpikeTrain(neurons={self.num_neurons}, active={self.count()})"


class SpikingNetwork:
    """Complete spiking neural network with LIF neurons and STDP learning."""

    def __init__(self, num_pre: int, num_post: int,
                 params: Optional[LIFNeuronParams] = None):
        self.num_pre = num_pre
        self.num_post = num_post
        self.params = params or LIFNeuronParams()

        # States
        self.pre_neurons = _snn.NeuronState(num_pre)
        self.post_neurons = _snn.NeuronState(num_post)
        self.pre_spikes = _snn.SpikeState(num_pre)
        self.post_spikes = _snn.SpikeState(num_post)

        # Synapses
        self.weights = _snn.SynapseMatrix(num_pre, num_post)
        self.weights.random_init(0.1, seed=42)

        # STDP traces
        self.trace_pre = np.zeros(num_pre, dtype=np.float32)
        self.trace_post = np.zeros(num_post, dtype=np.float32)

    def step(self, input_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation step. Returns (pre_spikes, post_spikes) as arrays."""
        input_pre = input_current[:self.num_pre].astype(np.float32)

        # Step pre-synaptic population
        _snn.step_lif(self.pre_neurons, self.pre_spikes,
                      self.weights, input_pre, self._params)

        # Compute post-synaptic input from pre spikes
        pre_arr = self.pre_spikes.to_array().astype(np.float32)
        post_input = pre_arr @ self.weights.get_weights()

        # Step post-synaptic population
        _snn.step_lif(self.post_neurons, self.post_spikes,
                      self.weights, post_input.astype(np.float32), self._params)

        return self.pre_spikes.to_array(), self.post_spikes.to_array()

    def stdp(self, a_plus=0.01, a_minus=0.012,
             tau_plus=20.0, tau_minus=20.0):
        """Apply STDP learning rule."""
        _snn.stdp_update(
            self.weights, self.pre_spikes, self.post_spikes,
            self.trace_pre, self.trace_post,
            a_plus, a_minus, tau_plus, tau_minus,
            self.params.dt
        )

    @property
    def weight_matrix(self) -> np.ndarray:
        return self.weights.get_weights()

    @weight_matrix.setter
    def weight_matrix(self, w: np.ndarray):
        self.weights.set_weights(w)


class CausalGraph:
    """Causal graph for event-driven reasoning."""

    def __init__(self, max_nodes: int = 1024, capacity: int = 1024):
        self._graph = _snn.CausalGraph(max_nodes, capacity)
        self.max_nodes = max_nodes

    def add_edge(self, pre: int, post: int,
                 weight: float = 1.0, credit: float = 0.0) -> int:
        return self._graph.add_edge(pre, post, weight, credit)

    def prune(self, threshold: float = 0.1) -> int:
        return self._graph.prune(threshold)

    def reinforce(self, active_nodes: np.ndarray, reward: float):
        self._graph.reinforce(active_nodes.astype(np.int32), reward)

    def propagate(self, source: int, steps: int = 3) -> np.ndarray:
        return self._graph.propagate(source, steps)

    @property
    def edge_count(self) -> int:
        return self._graph.edge_count()

    def outgoing(self, node: int) -> List[Tuple]:
        return self._graph.outgoing(node)

    def incoming(self, node: int) -> List[Tuple]:
        return self._graph.incoming(node)

    def __repr__(self):
        return f"CausalGraph(nodes={self.max_nodes}, edges={self.edge_count})"


class ThreeFactorLearner:
    """Three-factor learning with eligibility traces (Eq.18-20)."""

    def __init__(self, num_pre: int, num_post: int,
                 tau_e: float = 100.0):
        self.num_pre = num_pre
        self.num_post = num_post
        self.weights = _snn.SynapseMatrix(num_pre, num_post)
        self.weights.random_init(0.05, seed=42)
        self.eligibility = _snn.EligibilityTrace(num_pre, num_post, tau_e)
        self.tau_e = tau_e

    def update(self, pre_spikes: SpikeTrain, post_spikes: SpikeTrain,
               reward: float, eta: float = 0.001, dt: float = 1.0):
        """Apply three-factor weight update."""
        _snn.three_factor_update(
            self.weights, self.eligibility,
            pre_spikes._state, post_spikes._state,
            reward, eta, self.tau_e, dt
        )

    @property
    def weight_matrix(self) -> np.ndarray:
        return self.weights.get_weights()

    @property
    def eligibility_matrix(self) -> np.ndarray:
        return self.eligibility.get_traces()
