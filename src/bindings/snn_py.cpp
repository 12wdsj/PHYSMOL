#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/snn.h"
#include "../core/causal.h"

namespace py = pybind11;

PYBIND11_MODULE(_snn, m) {
    m.doc() = "PHYSMOL Spiking Neural Network - C core bindings";

    /* === LIFParams === */
    py::class_<LIFParams>(m, "LIFParams")
        .def(py::init([](float v_thresh, float v_reset, float tau_mem, float tau_ref, float dt) {
            auto *p = new LIFParams{v_thresh, v_reset, tau_mem, tau_ref, dt};
            return p;
        }),
        py::arg("v_threshold") = 1.0f,
        py::arg("v_reset") = 0.0f,
        py::arg("tau_mem") = 20.0f,
        py::arg("tau_ref") = 2.0f,
        py::arg("dt") = 1.0f)
        .def_readwrite("v_threshold", &LIFParams::v_threshold)
        .def_readwrite("v_reset", &LIFParams::v_reset)
        .def_readwrite("tau_mem", &LIFParams::tau_mem)
        .def_readwrite("tau_ref", &LIFParams::tau_ref)
        .def_readwrite("dt", &LIFParams::dt);

    /* === SpikeState === */
    py::class_<SpikeState>(m, "SpikeState")
        .def(py::init([](size_t num_neurons) {
            auto *s = snn_spike_create(num_neurons);
            if (!s) throw std::runtime_error("Failed to create SpikeState");
            return s;
        }))
        .def("clear", [](SpikeState &s) { snn_spike_clear(&s); })
        .def("count", [](const SpikeState &s) { return snn_spike_count(&s); })
        .def("to_array", [](const SpikeState &s) {
            py::array_t<int> arr({(py::ssize_t)s.num_neurons});
            snn_spike_to_array(&s, arr.mutable_data());
            return arr;
        })
        .def("from_array", [](SpikeState &s, py::array_t<int> arr) {
            auto buf = arr.request();
            if ((size_t)buf.shape[0] != s.num_neurons)
                throw std::runtime_error("Size mismatch");
            snn_spike_from_array(&s, (const int *)buf.ptr);
        })
        .def("set", [](SpikeState &s, size_t id) { snn_spike_set(&s, id); })
        .def("get", [](const SpikeState &s, size_t id) { return snn_spike_get(&s, id); })
        .def_readonly("num_neurons", &SpikeState::num_neurons);

    /* === NeuronState === */
    py::class_<NeuronState>(m, "NeuronState")
        .def(py::init([](size_t num_neurons) {
            auto *ns = snn_neuron_create(num_neurons);
            if (!ns) throw std::runtime_error("Failed to create NeuronState");
            return ns;
        }))
        .def("get_potentials", [](const NeuronState &ns) {
            py::array_t<float> arr({(py::ssize_t)ns.num_neurons});
            memcpy(arr.mutable_data(), ns.v_mem, ns.num_neurons * sizeof(float));
            return arr;
        })
        .def("set_potentials", [](NeuronState &ns, py::array_t<float> arr) {
            auto buf = arr.request();
            if ((size_t)buf.shape[0] != ns.num_neurons)
                throw std::runtime_error("Size mismatch");
            memcpy(ns.v_mem, buf.ptr, ns.num_neurons * sizeof(float));
        })
        .def_readonly("num_neurons", &NeuronState::num_neurons);

    /* === SynapseMatrix === */
    py::class_<SynapseMatrix>(m, "SynapseMatrix")
        .def(py::init([](size_t num_pre, size_t num_post) {
            auto *syn = snn_synapse_create(num_pre, num_post);
            if (!syn) throw std::runtime_error("Failed to create SynapseMatrix");
            return syn;
        }))
        .def("random_init", [](SynapseMatrix &syn, float scale, uint64_t seed) {
            snn_synapse_random_init(&syn, scale, seed);
        }, py::arg("scale") = 0.1f, py::arg("seed") = 0)
        .def("get_weights", [](const SynapseMatrix &syn) {
            return py::array_t<float>(
                {(py::ssize_t)syn.num_pre, (py::ssize_t)syn.num_post},
                {sizeof(float) * syn.num_post, sizeof(float)},
                syn.weights
            );
        })
        .def("set_weights", [](SynapseMatrix &syn, py::array_t<float> arr) {
            auto buf = arr.request();
            if ((size_t)buf.shape[0] != syn.num_pre || (size_t)buf.shape[1] != syn.num_post)
                throw std::runtime_error("Shape mismatch");
            memcpy(syn.weights, buf.ptr, syn.num_pre * syn.num_post * sizeof(float));
        })
        .def_readonly("num_pre", &SynapseMatrix::num_pre)
        .def_readonly("num_post", &SynapseMatrix::num_post);

    /* === EligibilityTrace === */
    py::class_<EligibilityTrace>(m, "EligibilityTrace")
        .def(py::init([](size_t num_pre, size_t num_post, float tau_e) {
            auto *et = snn_eligibility_create(num_pre, num_post, tau_e);
            if (!et) throw std::runtime_error("Failed to create EligibilityTrace");
            return et;
        }))
        .def("clear", [](EligibilityTrace &et) { snn_eligibility_clear(&et); })
        .def("get_traces", [](const EligibilityTrace &et) {
            return py::array_t<float>(
                {(py::ssize_t)et.num_pre, (py::ssize_t)et.num_post},
                {sizeof(float) * et.num_post, sizeof(float)},
                et.traces
            );
        })
        .def_readonly("tau_e", &EligibilityTrace::tau_e);

    /* === Core simulation functions === */

    m.def("step_lif", [](NeuronState &neurons, SpikeState &spikes,
                         const SynapseMatrix &weights,
                         py::array_t<float> input_current,
                         const LIFParams &params) {
        auto buf = input_current.request();
        if ((size_t)buf.shape[0] != neurons.num_neurons)
            throw std::runtime_error("Input current size mismatch");
        snn_step_lif(&neurons, &spikes, &weights, (float *)buf.ptr, &params);
    }, "Single LIF neuron simulation step");

    m.def("stdp_update", [](SynapseMatrix &weights,
                             const SpikeState &pre_spikes,
                             const SpikeState &post_spikes,
                             py::array_t<float> trace_pre,
                             py::array_t<float> trace_post,
                             float a_plus, float a_minus,
                             float tau_plus, float tau_minus,
                             float dt) {
        snn_stdp_update(&weights, &pre_spikes, &post_spikes,
                        trace_pre.mutable_data(), trace_post.mutable_data(),
                        a_plus, a_minus, tau_plus, tau_minus, dt);
    }, "STDP synaptic weight update");

    m.def("three_factor_update", [](SynapseMatrix &weights,
                                     EligibilityTrace &elig,
                                     const SpikeState &pre_spikes,
                                     const SpikeState &post_spikes,
                                     float reward, float eta, float tau_e, float dt) {
        snn_three_factor_update(&weights, &elig, &pre_spikes, &post_spikes,
                                reward, eta, tau_e, dt);
    }, "Three-factor learning rule (reward-modulated STDP)");

    /* === CausalGraph === */
    py::class_<CausalGraph>(m, "CausalGraph")
        .def(py::init([](size_t max_nodes, size_t capacity) {
            auto *g = causal_graph_create(max_nodes, capacity);
            if (!g) throw std::runtime_error("Failed to create CausalGraph");
            return g;
        }), py::arg("max_nodes"), py::arg("capacity") = 1024)
        .def("add_edge", [](CausalGraph &g, size_t pre, size_t post,
                            float weight, float credit) {
            return causal_graph_add_edge(&g, pre, post, weight, credit);
        }, py::arg("pre"), py::arg("post"), py::arg("weight") = 1.0f, py::arg("credit") = 0.0f)
        .def("prune", [](CausalGraph &g, float threshold) {
            return causal_graph_prune(&g, threshold);
        }, "Remove edges with credit below threshold")
        .def("reinforce", [](CausalGraph &g, py::array_t<int> active_nodes, float reward) {
            auto buf = active_nodes.request();
            causal_graph_reinforce(&g, (int *)buf.ptr, buf.shape[0], reward);
        }, "Reinforce edges between active nodes")
        .def("propagate", [](CausalGraph &g, size_t source, int steps) {
            py::array_t<float> activation({(py::ssize_t)g.max_nodes});
            auto buf = activation.request();
            memset(buf.ptr, 0, g.max_nodes * sizeof(float));
            ((float *)buf.ptr)[source] = 1.0f;
            causal_graph_propagate(&g, (float *)buf.ptr, source, steps);
            return activation;
        }, "Propagate activation from source node")
        .def("edge_count", [](const CausalGraph &g) {
            return causal_graph_edge_count(&g);
        })
        .def("outgoing", [](const CausalGraph &g, size_t node) {
            std::vector<CausalEdge> edges(g.num_edges);
            size_t count = causal_graph_outgoing(&g, node, edges.data(), edges.size());
            edges.resize(count);
            py::list result;
            for (auto &e : edges) {
                result.append(py::make_tuple(e.pre, e.post, e.weight, e.credit));
            }
            return result;
        })
        .def("incoming", [](const CausalGraph &g, size_t node) {
            std::vector<CausalEdge> edges(g.num_edges);
            size_t count = causal_graph_incoming(&g, node, edges.data(), edges.size());
            edges.resize(count);
            py::list result;
            for (auto &e : edges) {
                result.append(py::make_tuple(e.pre, e.post, e.weight, e.credit));
            }
            return result;
        })
        .def_readonly("max_nodes", &CausalGraph::max_nodes)
        .def_readonly("num_edges", &CausalGraph::num_edges);
}
