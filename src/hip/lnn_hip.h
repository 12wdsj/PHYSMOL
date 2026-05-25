#ifndef PHYSMOL_LNN_HIP_H
#define PHYSMOL_LNN_HIP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HIP-accelerated LNN forward pass.
 * These functions run on AMD GPU via ROCm/HIP.
 * Falls back to CPU (lnn.c) when HIP is unavailable.
 */

/* Check if HIP is available */
int lnn_hip_available(void);

/* Forward pass on GPU: compute L_θ(q, q̇) for a batch.
 * q, q_dot: device pointers, (batch_size * coord_dim)
 * L_out: device pointer, (batch_size)
 * Returns 0 on success, -1 on error.
 */
int lnn_forward_hip(const float *q, const float *q_dot,
                    const float *params,  /* flattened network params */
                    float *L_out,
                    size_t batch_size, size_t coord_dim, size_t hidden_dim);

/* Compute mass matrix (Hessian ∇²_{q̇} L) on GPU.
 * q_dot: device pointer, (coord_dim)
 * M_out: device pointer, (coord_dim * coord_dim)
 */
int lnn_mass_matrix_hip(const float *q, const float *q_dot,
                        const float *params,
                        float *M_out,
                        size_t coord_dim, size_t hidden_dim);

/* Batch VSA binding on GPU: out[i] = a[i] * b[i] (element-wise)
 * All arrays: (num_vectors * dim) on device.
 */
int vsa_bind_hip(float *out, const float *a, const float *b,
                 size_t num_vectors, size_t dim);

/* Batch cosine similarity on GPU.
 * Returns similarities array: (num_pairs) on host.
 */
int vsa_similarity_batch_hip(const float *a, const float *b,
                             float *sim_out,
                             size_t num_pairs, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_LNN_HIP_H */
