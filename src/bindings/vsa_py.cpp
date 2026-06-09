#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/vsa.h"

namespace py = pybind11;

/* Helper: wrap a raw float array as numpy (no copy) */
static py::array_t<float> vec_to_numpy(VSAVector *vec, bool owned = false) {
    if (!vec) throw std::runtime_error("Null VSAVector");
    /* Create numpy array that shares memory with VSAVector */
    py::array_t<float> arr({(py::ssize_t)vec->dim}, {sizeof(float)}, vec->data);
    if (owned) {
        /* Store pointer in capsule for cleanup */
        py::capsule cap(vec, [](void *v) {
            vsa_free(static_cast<VSAVector *>(v));
        });
        arr.attr("_owner") = cap;
    }
    return arr;
}

PYBIND11_MODULE(_vsa, m) {
    m.doc() = "PHYSMOL Vector Symbolic Architecture - C core bindings";

    /* === VSAVector operations (operate on numpy arrays directly) === */

    m.def("bind", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Inputs must be 1D arrays");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Dimension mismatch");

        size_t dim = buf_a.shape[0];
        auto result = py::array_t<float>({(py::ssize_t)dim});

        VSAVector va = {(float *)buf_a.ptr, dim};
        VSAVector vb = {(float *)buf_b.ptr, dim};
        VSAVector vout = {result.mutable_data(), dim};

        vsa_bind(&vout, &va, &vb);
        return result;
    }, "Binding (Hadamard product): out = a * b");

    m.def("bundle", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request(), buf_b = b.request();
        if (buf_a.ndim != 1 || buf_b.ndim != 1)
            throw std::runtime_error("Inputs must be 1D arrays");
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Dimension mismatch");

        size_t dim = buf_a.shape[0];
        auto result = py::array_t<float>({(py::ssize_t)dim});

        VSAVector va = {(float *)buf_a.ptr, dim};
        VSAVector vb = {(float *)buf_b.ptr, dim};
        VSAVector vout = {result.mutable_data(), dim};

        vsa_bundle(&vout, &va, &vb);
        return result;
    }, "Bundling (addition): out = a + b");

    m.def("unbind", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request(), buf_b = b.request();
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Dimension mismatch");

        size_t dim = buf_a.shape[0];
        auto result = py::array_t<float>({(py::ssize_t)dim});

        VSAVector va = {(float *)buf_a.ptr, dim};
        VSAVector vb = {(float *)buf_b.ptr, dim};
        VSAVector vout = {result.mutable_data(), dim};

        vsa_unbind(&vout, &va, &vb);
        return result;
    }, "Unbinding: out = a * inverse(b)");

    m.def("cosine_similarity", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request(), buf_b = b.request();
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Dimension mismatch");

        size_t dim = buf_a.shape[0];
        VSAVector va = {(float *)buf_a.ptr, dim};
        VSAVector vb = {(float *)buf_b.ptr, dim};
        return vsa_cosine_similarity(&va, &vb);
    }, "Cosine similarity between two vectors");

    m.def("hamming_distance", [](py::array_t<float> a, py::array_t<float> b) {
        auto buf_a = a.request(), buf_b = b.request();
        if (buf_a.shape[0] != buf_b.shape[0])
            throw std::runtime_error("Dimension mismatch");

        size_t dim = buf_a.shape[0];
        VSAVector va = {(float *)buf_a.ptr, dim};
        VSAVector vb = {(float *)buf_b.ptr, dim};
        return vsa_hamming_distance(&va, &vb);
    }, "Normalized Hamming distance for bipolar vectors");

    m.def("permute", [](py::array_t<float> vec, int shift) {
        auto buf = vec.request();
        size_t dim = buf.shape[0];
        auto result = py::array_t<float>({(py::ssize_t)dim});

        VSAVector vin = {(float *)buf.ptr, dim};
        VSAVector vout = {result.mutable_data(), dim};

        vsa_permute(&vout, &vin, shift);
        return result;
    }, "Circular permutation (shift)");

    m.def("random_bipolar", [](size_t dim, uint64_t seed) {
        auto result = py::array_t<float>({(py::ssize_t)dim});
        VSAVector vout = {result.mutable_data(), dim};
        vsa_random_bipolar(&vout, seed);
        return result;
    }, "Generate random bipolar vector (-1/+1)",
       py::arg("dim"), py::arg("seed") = 0);

    m.def("random_phase", [](size_t dim, uint64_t seed) {
        auto result = py::array_t<float>({(py::ssize_t)dim});
        VSAVector vout = {result.mutable_data(), dim};
        vsa_random_phase(&vout, seed);
        return result;
    }, "Generate random phase vector (for FHRR)",
       py::arg("dim"), py::arg("seed") = 0);

    m.def("fpe_encode", [](float x, float y, float z,
                            py::array_t<float> base_x,
                            py::array_t<float> base_y,
                            py::array_t<float> base_z) {
        auto bx = base_x.request(), by = base_y.request(), bz = base_z.request();
        size_t dim = bx.shape[0];
        auto result = py::array_t<float>({(py::ssize_t)dim});

        VSAVector vbx = {(float *)bx.ptr, dim};
        VSAVector vby = {(float *)by.ptr, dim};
        VSAVector vbz = {(float *)bz.ptr, dim};
        VSAVector vout = {result.mutable_data(), dim};

        vsa_fpe_encode(&vout, x, y, z, &vbx, &vby, &vbz);
        return result;
    }, "FPE encode 3D coordinate into hypervector");

    m.def("normalize", [](py::array_t<float> vec) {
        auto buf = vec.request();
        VSAVector v = {(float *)buf.ptr, (size_t)buf.shape[0]};
        vsa_normalize(&v);
    }, "Normalize vector to unit length (in-place)");

    m.def("quantize", [](py::array_t<float> vec, int bits) {
        auto buf = vec.request();
        VSAVector v = {(float *)buf.ptr, (size_t)buf.shape[0]};
        vsa_quantize_inplace(&v, bits);
    }, "Quantize vector in-place",
       py::arg("vec"), py::arg("bits") = 8);

    /* === Codebook === */

    py::class_<VSACodebook>(m, "Codebook")
        .def(py::init([](size_t dim, size_t capacity) {
            auto *cb = vsa_codebook_create(dim, capacity);
            if (!cb) throw std::runtime_error("Failed to create codebook");
            return cb;
        }), py::arg("dim"), py::arg("capacity") = 256)
        .def("add", [](VSACodebook &cb, const std::string &name, py::array_t<float> vec) {
            auto buf = vec.request();
            if ((size_t)buf.shape[0] != cb.dim)
                throw std::runtime_error("Dimension mismatch");
            VSAVector v = {(float *)buf.ptr, cb.dim};
            size_t idx = vsa_codebook_add(&cb, name.c_str(), &v);
            if (idx == (size_t)-1) throw std::runtime_error("Failed to add to codebook");
            return idx;
        }, "Add a named primitive vector")
        .def("lookup", [](VSACodebook &cb, const std::string &name) -> py::object {
            const VSAVector *v = vsa_codebook_lookup(&cb, name.c_str());
            if (!v) return py::none();
            /* Return a copy (the internal pointer is static/thread-local) */
            auto result = py::array_t<float>({(py::ssize_t)cb.dim});
            memcpy(result.mutable_data(), v->data, cb.dim * sizeof(float));
            return result;
        }, "Look up a primitive by name, returns numpy array or None")
        .def("nearest", [](VSACodebook &cb, py::array_t<float> query) {
            auto buf = query.request();
            if ((size_t)buf.shape[0] != cb.dim)
                throw std::runtime_error("Dimension mismatch");
            VSAVector vq = {(float *)buf.ptr, cb.dim};
            float sim;
            size_t idx = vsa_codebook_nearest(&cb, &vq, &sim);
            return py::make_tuple(idx, sim, std::string(cb.names[idx]));
        }, "Find most similar vector, returns (index, similarity, name)")
        .def("__len__", [](const VSACodebook &cb) { return cb.num_vectors; })
        .def_readonly("dim", &VSACodebook::dim)
        .def_readonly("num_vectors", &VSACodebook::num_vectors);
}
