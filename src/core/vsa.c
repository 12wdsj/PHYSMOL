#include "vsa.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* SIMD detection */
#if defined(__AVX2__) && defined(__FMA__)
    #include <immintrin.h>
    #define VSA_USE_AVX2 1
#else
    #define VSA_USE_AVX2 0
#endif

/* Simple xorshift64 PRNG */
static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* === Lifecycle === */

VSAVector *vsa_create(size_t dim) {
    VSAVector *vec = (VSAVector *)malloc(sizeof(VSAVector));
    if (!vec) return NULL;
    vec->data = (float *)calloc(dim, sizeof(float));
    if (!vec->data) { free(vec); return NULL; }
    vec->dim = dim;
    return vec;
}

void vsa_free(VSAVector *vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

VSAVector *vsa_clone(const VSAVector *src) {
    VSAVector *dst = vsa_create(src->dim);
    if (dst) memcpy(dst->data, src->data, src->dim * sizeof(float));
    return dst;
}

/* === Random generation === */

void vsa_random_bipolar(VSAVector *out, uint64_t seed) {
    size_t dim = out->dim;
    uint64_t state = seed ? seed : 42;
    for (size_t i = 0; i < dim; i++) {
        uint64_t r = xorshift64(&state);
        out->data[i] = (r & 1) ? 1.0f : -1.0f;
    }
}

void vsa_random_phase(VSAVector *out, uint64_t seed) {
    size_t dim = out->dim;
    uint64_t state = seed ? seed : 42;
    for (size_t i = 0; i < dim; i++) {
        /* Generate random phase in [0, 2*pi) */
        uint64_t r = xorshift64(&state);
        double phase = (double)(r & 0xFFFFFFFF) / (double)0xFFFFFFFF * 6.283185307179586;
        out->data[i] = (float)phase;
    }
}

/* === Core algebraic operations === */

void vsa_bind(VSAVector *out, const VSAVector *a, const VSAVector *b) {
    size_t dim = a->dim;

#if VSA_USE_AVX2
    size_t i = 0;
    size_t simd_end = dim & ~7u; /* round down to multiple of 8 */
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&out->data[i], vr);
    }
    for (; i < dim; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
#endif
}

void vsa_bundle(VSAVector *out, const VSAVector *a, const VSAVector *b) {
    size_t dim = a->dim;

#if VSA_USE_AVX2
    size_t i = 0;
    size_t simd_end = dim & ~7u;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&out->data[i], vr);
    }
    for (; i < dim; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
#endif
}

void vsa_unbind(VSAVector *out, const VSAVector *a, const VSAVector *b) {
    /* For bipolar vectors: inverse(b) = b, so unbind = bind */
    /* For FHRR (phase): inverse = conjugate = negate phase */
    /* General case: element-wise divide (approximate inverse) */
    size_t dim = a->dim;

#if VSA_USE_AVX2
    size_t i = 0;
    size_t simd_end = dim & ~7u;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&out->data[i], vr);
    }
    for (; i < dim; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
#endif
}

void vsa_permute(VSAVector *out, const VSAVector *in, int shift) {
    size_t dim = in->dim;
    /* Normalize shift to [0, dim) */
    shift = shift % (int)dim;
    if (shift < 0) shift += (int)dim;

    for (size_t i = 0; i < dim; i++) {
        size_t src = (i + (size_t)shift) % dim;
        out->data[i] = in->data[src];
    }
}

/* === Similarity === */

float vsa_cosine_similarity(const VSAVector *a, const VSAVector *b) {
    size_t dim = a->dim;
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

#if VSA_USE_AVX2
    __m256 vdot = _mm256_setzero_ps();
    __m256 vna  = _mm256_setzero_ps();
    __m256 vnb  = _mm256_setzero_ps();

    size_t i = 0;
    size_t simd_end = dim & ~7u;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a->data[i]);
        __m256 vb = _mm256_loadu_ps(&b->data[i]);
        vdot = _mm256_fmadd_ps(va, vb, vdot);  /* FMA: a*b + dot */
        vna  = _mm256_fmadd_ps(va, va, vna);
        vnb  = _mm256_fmadd_ps(vb, vb, vnb);
    }
    /* Horizontal sum */
    float dot_arr[8], na_arr[8], nb_arr[8];
    _mm256_storeu_ps(dot_arr, vdot);
    _mm256_storeu_ps(na_arr, vna);
    _mm256_storeu_ps(nb_arr, vnb);
    for (int j = 0; j < 8; j++) {
        dot += dot_arr[j]; norm_a += na_arr[j]; norm_b += nb_arr[j];
    }
    for (; i < dim; i++) {
        dot += a->data[i] * b->data[i];
        norm_a += a->data[i] * a->data[i];
        norm_b += b->data[i] * b->data[i];
    }
#else
    for (size_t i = 0; i < dim; i++) {
        dot += a->data[i] * b->data[i];
        norm_a += a->data[i] * a->data[i];
        norm_b += b->data[i] * b->data[i];
    }
#endif

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

float vsa_hamming_distance(const VSAVector *a, const VSAVector *b) {
    size_t dim = a->dim;
    size_t diff = 0;
    for (size_t i = 0; i < dim; i++) {
        if ((a->data[i] > 0) != (b->data[i] > 0)) diff++;
    }
    return (float)diff / (float)dim;
}

/* === FPE encoding === */

void vsa_fpe_encode(VSAVector *out,
                    float x, float y, float z,
                    const VSAVector *base_x,
                    const VSAVector *base_y,
                    const VSAVector *base_z) {
    size_t dim = out->dim;

    /* H(s) = B_x^x * B_y^y * B_z^z
     * In FHRR, this means: phase_j = x*phase(Bx_j) + y*phase(By_j) + z*phase(Bz_j)
     * Result is element-wise: exp(i * combined_phase)
     * For float32 representation, we store the combined phase values
     */
    for (size_t i = 0; i < dim; i++) {
        float combined_phase = x * base_x->data[i] + y * base_y->data[i] + z * base_z->data[i];
        out->data[i] = combined_phase;
    }
}

/* === Quantization === */

void vsa_quantize_inplace(VSAVector *vec, int bits) {
    size_t dim = vec->dim;
    if (bits <= 0 || bits >= 32) return; /* no-op for full precision */

    /* Find range */
    float min_val = vec->data[0], max_val = vec->data[0];
    for (size_t i = 1; i < dim; i++) {
        if (vec->data[i] < min_val) min_val = vec->data[i];
        if (vec->data[i] > max_val) max_val = vec->data[i];
    }

    float range = max_val - min_val;
    if (range < 1e-10f) return;

    int levels = (1 << bits) - 1; /* e.g., 255 for 8-bit */
    float scale = (float)levels / range;

    for (size_t i = 0; i < dim; i++) {
        int q = (int)((vec->data[i] - min_val) * scale + 0.5f);
        if (q < 0) q = 0;
        if (q > levels) q = levels;
        vec->data[i] = (float)q / scale + min_val;
    }
}

void vsa_normalize(VSAVector *vec) {
    size_t dim = vec->dim;
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        norm_sq += vec->data[i] * vec->data[i];
    }
    float norm = sqrtf(norm_sq);
    if (norm < 1e-10f) return;
    for (size_t i = 0; i < dim; i++) {
        vec->data[i] /= norm;
    }
}

/* === Codebook === */

VSACodebook *vsa_codebook_create(size_t dim, size_t initial_capacity) {
    VSACodebook *cb = (VSACodebook *)calloc(1, sizeof(VSACodebook));
    if (!cb) return NULL;
    cb->dim = dim;
    cb->capacity = initial_capacity ? initial_capacity : 256;
    cb->num_vectors = 0;
    cb->vectors = (float *)calloc(cb->capacity * dim, sizeof(float));
    cb->names = (char **)calloc(cb->capacity, sizeof(char *));
    if (!cb->vectors || !cb->names) {
        vsa_codebook_free(cb);
        return NULL;
    }
    return cb;
}

void vsa_codebook_free(VSACodebook *cb) {
    if (!cb) return;
    for (size_t i = 0; i < cb->num_vectors; i++) {
        free(cb->names[i]);
    }
    free(cb->vectors);
    free(cb->names);
    free(cb);
}

size_t vsa_codebook_add(VSACodebook *cb, const char *name, const VSAVector *vec) {
    if (cb->num_vectors >= cb->capacity) {
        /* Grow */
        size_t new_cap = cb->capacity * 2;
        float *new_vecs = (float *)realloc(cb->vectors, new_cap * cb->dim * sizeof(float));
        char **new_names = (char **)realloc(cb->names, new_cap * sizeof(char *));
        if (!new_vecs || !new_names) return (size_t)-1;
        cb->vectors = new_vecs;
        cb->names = new_names;
        cb->capacity = new_cap;
    }
    size_t idx = cb->num_vectors;
    memcpy(&cb->vectors[idx * cb->dim], vec->data, cb->dim * sizeof(float));
    cb->names[idx] = strdup(name);
    cb->num_vectors++;
    return idx;
}

const VSAVector *vsa_codebook_lookup(const VSACodebook *cb, const char *name) {
    /* Temporary vector wrapper for returning */
    for (size_t i = 0; i < cb->num_vectors; i++) {
        if (strcmp(cb->names[i], name) == 0) {
            /* We need to return a VSAVector*, but we store flat data.
             * Caller should use vsa_codebook_nearest or access directly.
             * For safety, we use a thread-local static wrapper. */
            static __declspec(thread) VSAVector wrapper;
            wrapper.data = (float *)&cb->vectors[i * cb->dim];
            wrapper.dim = cb->dim;
            return &wrapper;
        }
    }
    return NULL;
}

size_t vsa_codebook_nearest(const VSACodebook *cb, const VSAVector *query, float *sim_out) {
    float best_sim = -2.0f;
    size_t best_idx = 0;

    VSAVector candidate;
    candidate.dim = cb->dim;

    for (size_t i = 0; i < cb->num_vectors; i++) {
        candidate.data = (float *)&cb->vectors[i * cb->dim];
        float sim = vsa_cosine_similarity(query, &candidate);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    if (sim_out) *sim_out = best_sim;
    return best_idx;
}
