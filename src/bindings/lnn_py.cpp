#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../core/lnn.h"
#include "../core/memory.h"

namespace py = pybind11;

PYBIND11_MODULE(_lnn, m) {
    m.doc() = "PHYSMOL Lagrangian Neural Network & Memory - C core bindings";

    /* === LNNParams === */
    py::class_<LNNParams>(m, "LNNParams")
        .def(py::init([](size_t coord_dim, size_t hidden_dim) {
            auto *p = lnn_create(coord_dim, hidden_dim);
            if (!p) throw std::runtime_error("Failed to create LNNParams");
            return p;
        }))
        .def("random_init", [](LNNParams &p, uint64_t seed) {
            lnn_random_init(&p, seed);
        }, py::arg("seed") = 42)
        .def("forward", [](const LNNParams &p,
                           py::array_t<float> q, py::array_t<float> q_dot) {
            auto bq = q.request(), bqd = q_dot.request();
            if ((size_t)bq.shape[0] != p.coord_dim)
                throw std::runtime_error("q dimension mismatch");
            if ((size_t)bqd.shape[0] != p.coord_dim)
                throw std::runtime_error("q_dot dimension mismatch");
            return lnn_forward(&p, (float *)bq.ptr, (float *)bqd.ptr);
        }, "Compute Lagrangian L_θ(q, q̇)")
        .def("forward_batch", [](const LNNParams &p,
                                  py::array_t<float> q, py::array_t<float> q_dot) {
            auto bq = q.request(), bqd = q_dot.request();
            size_t batch = bq.shape[0];
            size_t cd = p.coord_dim;
            if ((size_t)bq.shape[1] != cd || (size_t)bqd.shape[1] != cd)
                throw std::runtime_error("Dimension mismatch");

            py::array_t<float> L_out({(py::ssize_t)batch});
            lnn_forward_batch(&p, (float *)bq.ptr, (float *)bqd.ptr,
                             L_out.mutable_data(), batch);
            return L_out;
        }, "Batch forward: returns array of Lagrangian values")
        .def("compute_acceleration", [](const LNNParams &p,
                                         py::array_t<float> q,
                                         py::array_t<float> q_dot) {
            auto bq = q.request(), bqd = q_dot.request();
            if ((size_t)bq.shape[0] != p.coord_dim)
                throw std::runtime_error("q dimension mismatch");

            py::array_t<float> q_ddot({(py::ssize_t)p.coord_dim});
            lnn_compute_acceleration(&p, (float *)bq.ptr, (float *)bqd.ptr,
                                     q_ddot.mutable_data());
            return q_ddot;
        }, "Compute q̈ from Euler-Lagrange equation (CPU fallback)")
        .def("gradient", [](const LNNParams &p,
                            py::array_t<float> q, py::array_t<float> q_dot) {
            auto bq = q.request(), bqd = q_dot.request();
            size_t cd = p.coord_dim;
            py::array_t<float> dL_dq({(py::ssize_t)cd});
            py::array_t<float> dL_dqdot({(py::ssize_t)cd});
            lnn_gradient(&p, (float *)bq.ptr, (float *)bqd.ptr,
                        dL_dq.mutable_data(), dL_dqdot.mutable_data());
            return py::make_tuple(dL_dq, dL_dqdot);
        }, "Compute ∂L/∂q and ∂L/∂q̇ via finite differences")
        .def_readonly("coord_dim", &LNNParams::coord_dim)
        .def_readonly("hidden_dim", &LNNParams::hidden_dim)
        .def_readonly("input_dim", &LNNParams::input_dim);

    /* === TieredMemory === */
    py::class_<TieredMemory>(m, "TieredMemory")
        .def(py::init([](size_t l1_capacity, size_t l2_buckets, const std::string &l3_path) {
            auto *mem = tiered_mem_create(l1_capacity, l2_buckets,
                                          l3_path.empty() ? nullptr : l3_path.c_str());
            if (!mem) throw std::runtime_error("Failed to create TieredMemory");
            return mem;
        }), py::arg("l1_capacity") = 1024, py::arg("l2_buckets") = 4096,
            py::arg("l3_path") = "")
        .def("put_l1", [](TieredMemory &mem, uint64_t key, py::array_t<float> data) {
            auto buf = data.request();
            return tiered_mem_l1_put(&mem, key, (float *)buf.ptr, buf.shape[0]);
        })
        .def("put_l2", [](TieredMemory &mem, uint64_t key, py::array_t<float> data) {
            auto buf = data.request();
            return tiered_mem_l2_put(&mem, key, (float *)buf.ptr, buf.shape[0]);
        })
        .def("get", [](TieredMemory &mem, uint64_t key) -> py::object {
            size_t size;
            float *data = tiered_mem_get(&mem, key, &size);
            if (!data) return py::none();
            py::array_t<float> arr({(py::ssize_t)size});
            memcpy(arr.mutable_data(), data, size * sizeof(float));
            return arr;
        }, "Get with auto-promotion: L1 -> L2 -> L3")
        .def("stats", [](const TieredMemory &mem) {
            size_t l1c, l2c, hits, misses;
            tiered_mem_stats(&mem, &l1c, &l2c, &hits, &misses);
            return py::dict(
                "l1_count"_a = l1c, "l2_count"_a = l2c,
                "hits"_a = hits, "misses"_a = misses
            );
        });
}
