#ifndef PHYSMOL_VSA_H
#define PHYSMOL_VSA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Default VSA dimension */
#define VSA_DEFAULT_DIM 4096

/* VSA vector: flat float32 array of length `dim` */
typedef struct {
    float *data;
    size_t dim;
} VSAVector;

/* Codebook: collection of named primitive vectors */
typedef struct {
    float *vectors;      /* flat array: num_vectors * dim */
    char **names;        /* name for each vector */
    size_t num_vectors;
    size_t dim;
    size_t capacity;
} VSACodebook;

/* === Lifecycle === */

/* Allocate a zeroed vector */
VSAVector *vsa_create(size_t dim);

/* Free a vector */
void vsa_free(VSAVector *vec);

/* Clone a vector */
VSAVector *vsa_clone(const VSAVector *src);

/* === Random generation === */

/* Fill with random bipolar values (-1/+1) */
void vsa_random_bipolar(VSAVector *out, uint64_t seed);

/* Fill with random values from unit circle (complex phases) for FHRR */
void vsa_random_phase(VSAVector *out, uint64_t seed);

/* === Core algebraic operations (SIMD-optimized) === */

/* Binding: out = a * b (Hadamard product, element-wise multiply) */
void vsa_bind(VSAVector *out, const VSAVector *a, const VSAVector *b);

/* Bundling: out = a + b (element-wise addition) */
void vsa_bundle(VSAVector *out, const VSAVector *a, const VSAVector *b);

/* Unbinding: out = a * inverse(b). For bipolar: inverse = identity */
void vsa_unbind(VSAVector *out, const VSAVector *a, const VSAVector *b);

/* Permutation: circular shift (for positional encoding) */
void vsa_permute(VSAVector *out, const VSAVector *in, int shift);

/* === Similarity === */

/* Cosine similarity between two vectors */
float vsa_cosine_similarity(const VSAVector *a, const VSAVector *b);

/* Hamming distance for bipolar vectors */
float vsa_hamming_distance(const VSAVector *a, const VSAVector *b);

/* === FPE (Fractional Power Encoding) for spatial coordinates === */

/* Encode a 3D coordinate into a hypervector using FPE:
 * H(s) = B_x^x * B_y^y * B_z^z
 * base_x/y/z should be random phase vectors
 */
void vsa_fpe_encode(VSAVector *out,
                    float x, float y, float z,
                    const VSAVector *base_x,
                    const VSAVector *base_y,
                    const VSAVector *base_z);

/* === Quantization === */

/* In-place quantization to specified bit width (reduces memory for storage) */
void vsa_quantize_inplace(VSAVector *vec, int bits);

/* Normalize vector to unit length */
void vsa_normalize(VSAVector *vec);

/* === Codebook operations === */

VSACodebook *vsa_codebook_create(size_t dim, size_t initial_capacity);
void vsa_codebook_free(VSACodebook *cb);

/* Add a named primitive to the codebook. Returns its index. */
size_t vsa_codebook_add(VSACodebook *cb, const char *name, const VSAVector *vec);

/* Look up a primitive by name. Returns NULL if not found. */
const VSAVector *vsa_codebook_lookup(const VSACodebook *cb, const char *name);

/* Find the most similar vector in the codebook. Returns index. */
size_t vsa_codebook_nearest(const VSACodebook *cb, const VSAVector *query, float *sim_out);

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_VSA_H */
