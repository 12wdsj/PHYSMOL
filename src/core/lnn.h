#ifndef PHYSMOL_LNN_H
#define PHYSMOL_LNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LNN parameters: MLP weights for Lagrangian L_θ(q, q̇) */
typedef struct {
    /* Layer 1: input -> hidden */
    float *w1;   /* (input_dim * hidden_dim) */
    float *b1;   /* hidden_dim */
    /* Layer 2: hidden -> hidden */
    float *w2;   /* (hidden_dim * hidden_dim) */
    float *b2;   /* hidden_dim */
    /* Layer 3: hidden -> 1 (scalar Lagrangian) */
    float *w3;   /* hidden_dim */
    float b3;

    size_t input_dim;   /* 2 * coord_dim (q concatenated with q̇) */
    size_t hidden_dim;
    size_t coord_dim;   /* generalized coordinates dimension */
} LNNParams;

/* === Lifecycle === */
LNNParams *lnn_create(size_t coord_dim, size_t hidden_dim);
void lnn_free(LNNParams *params);

/* Random initialization (Xavier) */
void lnn_random_init(LNNParams *params, uint64_t seed);

/* === Forward pass (CPU) === */

/* Compute Lagrangian L_θ(q, q̇) for a single sample.
 * q and q_dot are arrays of length coord_dim.
 * Returns scalar Lagrangian value.
 */
float lnn_forward(const LNNParams *params, const float *q, const float *q_dot);

/* Compute Lagrangian for a batch.
 * q: (batch_size * coord_dim), q_dot: same.
 * L_out: (batch_size) output values.
 */
void lnn_forward_batch(const LNNParams *params,
                       const float *q, const float *q_dot,
                       float *L_out, size_t batch_size);

/* === Gradient computation (CPU) === */

/* Compute ∂L/∂q and ∂L/∂q̇ via finite differences (for CPU fallback).
 * In practice, PyTorch autograd is preferred.
 */
void lnn_gradient(const LNNParams *params, const float *q, const float *q_dot,
                  float *dL_dq, float *dL_dqdot);

/* Compute acceleration from Euler-Lagrange equation (Eq.16):
 * q̈ = (∇²_{q̇} L)⁻¹ [∇_q L - (∇_q ∇_{q̇} L) q̇]
 * Uses finite differences for Hessian approximation.
 */
void lnn_compute_acceleration(const LNNParams *params,
                              const float *q, const float *q_dot,
                              float *q_ddot);

/* === Neural network internals (for debug/inspection) === */

/* ReLU activation */
static inline float relu(float x) { return x > 0 ? x : 0; }

/* MLP forward: given input x (input_dim), compute output (1D) through 3 layers */
float lnn_mlp_forward(const LNNParams *params, const float *x);

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_LNN_H */
