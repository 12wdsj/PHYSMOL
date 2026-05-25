#include "lnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Simple xorshift64 */
static uint64_t _lnn_xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

LNNParams *lnn_create(size_t coord_dim, size_t hidden_dim) {
    LNNParams *p = (LNNParams *)calloc(1, sizeof(LNNParams));
    if (!p) return NULL;

    p->coord_dim = coord_dim;
    p->hidden_dim = hidden_dim;
    p->input_dim = 2 * coord_dim;  /* [q, q̇] concatenated */

    size_t in = p->input_dim, h = hidden_dim;

    p->w1 = (float *)calloc(in * h, sizeof(float));
    p->b1 = (float *)calloc(h, sizeof(float));
    p->w2 = (float *)calloc(h * h, sizeof(float));
    p->b2 = (float *)calloc(h, sizeof(float));
    p->w3 = (float *)calloc(h, sizeof(float));
    p->b3 = 0.0f;

    if (!p->w1 || !p->b1 || !p->w2 || !p->b2 || !p->w3) {
        lnn_free(p);
        return NULL;
    }
    return p;
}

void lnn_free(LNNParams *params) {
    if (!params) return;
    free(params->w1); free(params->b1);
    free(params->w2); free(params->b2);
    free(params->w3);
    free(params);
}

void lnn_random_init(LNNParams *params, uint64_t seed) {
    uint64_t rng = seed ? seed : 42;
    size_t in = params->input_dim, h = params->hidden_dim;

    /* Xavier initialization: scale = sqrt(2 / (fan_in + fan_out)) */

    /* Layer 1 */
    float scale1 = sqrtf(2.0f / (float)(in + h));
    for (size_t i = 0; i < in * h; i++) {
        uint64_t r = _lnn_xorshift64(&rng);
        double u = (double)(r & 0xFFFFFFFF) / (double)0xFFFFFFFF;
        params->w1[i] = (float)((u - 0.5) * 2.0 * scale1);
    }

    /* Layer 2 */
    float scale2 = sqrtf(2.0f / (float)(h + h));
    for (size_t i = 0; i < h * h; i++) {
        uint64_t r = _lnn_xorshift64(&rng);
        double u = (double)(r & 0xFFFFFFFF) / (double)0xFFFFFFFF;
        params->w2[i] = (float)((u - 0.5) * 2.0 * scale2);
    }

    /* Layer 3 */
    float scale3 = sqrtf(2.0f / (float)(h + 1));
    for (size_t i = 0; i < h; i++) {
        uint64_t r = _lnn_xorshift64(&rng);
        double u = (double)(r & 0xFFFFFFFF) / (double)0xFFFFFFFF;
        params->w3[i] = (float)((u - 0.5) * 2.0 * scale3);
    }
}

float lnn_mlp_forward(const LNNParams *params, const float *x) {
    size_t in = params->input_dim, h = params->hidden_dim;

    /* Layer 1: h1 = relu(W1 @ x + b1) */
    float *h1 = (float *)alloca(h * sizeof(float));
    for (size_t j = 0; j < h; j++) {
        float sum = params->b1[j];
        for (size_t i = 0; i < in; i++) {
            sum += params->w1[i * h + j] * x[i];
        }
        h1[j] = relu(sum);
    }

    /* Layer 2: h2 = relu(W2 @ h1 + b2) */
    float *h2 = (float *)alloca(h * sizeof(float));
    for (size_t j = 0; j < h; j++) {
        float sum = params->b2[j];
        for (size_t i = 0; i < h; i++) {
            sum += params->w2[i * h + j] * h1[i];
        }
        h2[j] = relu(sum);
    }

    /* Layer 3: L = W3 @ h2 + b3 */
    float L = params->b3;
    for (size_t i = 0; i < h; i++) {
        L += params->w3[i] * h2[i];
    }

    return L;
}

float lnn_forward(const LNNParams *params, const float *q, const float *q_dot) {
    size_t cd = params->coord_dim;
    /* Concatenate [q, q̇] */
    float *x = (float *)alloca(2 * cd * sizeof(float));
    memcpy(x, q, cd * sizeof(float));
    memcpy(x + cd, q_dot, cd * sizeof(float));

    return lnn_mlp_forward(params, x);
}

void lnn_forward_batch(const LNNParams *params,
                       const float *q, const float *q_dot,
                       float *L_out, size_t batch_size) {
    size_t cd = params->coord_dim;
    for (size_t b = 0; b < batch_size; b++) {
        L_out[b] = lnn_forward(params, &q[b * cd], &q_dot[b * cd]);
    }
}

void lnn_gradient(const LNNParams *params, const float *q, const float *q_dot,
                  float *dL_dq, float *dL_dqdot) {
    size_t cd = params->coord_dim;
    float eps = 1e-4f;

    /* ∂L/∂q via finite differences */
    for (size_t i = 0; i < cd; i++) {
        float q_plus[cd], q_minus[cd];
        memcpy(q_plus, q, cd * sizeof(float));
        memcpy(q_minus, q, cd * sizeof(float));
        q_plus[i] += eps;
        q_minus[i] -= eps;

        float L_plus = lnn_forward(params, q_plus, q_dot);
        float L_minus = lnn_forward(params, q_minus, q_dot);
        dL_dq[i] = (L_plus - L_minus) / (2.0f * eps);
    }

    /* ∂L/∂q̇ via finite differences */
    for (size_t i = 0; i < cd; i++) {
        float qd_plus[cd], qd_minus[cd];
        memcpy(qd_plus, q_dot, cd * sizeof(float));
        memcpy(qd_minus, q_dot, cd * sizeof(float));
        qd_plus[i] += eps;
        qd_minus[i] -= eps;

        float L_plus = lnn_forward(params, q, qd_plus);
        float L_minus = lnn_forward(params, q, qd_minus);
        dL_dqdot[i] = (L_plus - L_minus) / (2.0f * eps);
    }
}

void lnn_compute_acceleration(const LNNParams *params,
                              const float *q, const float *q_dot,
                              float *q_ddot) {
    size_t cd = params->coord_dim;
    float eps = 1e-4f;

    /* Compute ∂L/∂q and ∂L/∂q̇ */
    float *dL_dq = (float *)alloca(cd * sizeof(float));
    float *dL_dqdot = (float *)alloca(cd * sizeof(float));
    lnn_gradient(params, q, q_dot, dL_dq, dL_dqdot);

    /* Approximate Hessian ∇²_{q̇} L via finite differences */
    /* H_ij = ∂²L / ∂q̇_i ∂q̇_j */
    float *H = (float *)alloca(cd * cd * sizeof(float));
    for (size_t i = 0; i < cd; i++) {
        for (size_t j = 0; j < cd; j++) {
            float qd_pp[cd], qd_pm[cd], qd_mp[cd], qd_mm[cd];
            memcpy(qd_pp, q_dot, cd * sizeof(float));
            memcpy(qd_pm, q_dot, cd * sizeof(float));
            memcpy(qd_mp, q_dot, cd * sizeof(float));
            memcpy(qd_mm, q_dot, cd * sizeof(float));

            qd_pp[i] += eps; qd_pp[j] += eps;
            qd_pm[i] += eps; qd_pm[j] -= eps;
            qd_mp[i] -= eps; qd_mp[j] += eps;
            qd_mm[i] -= eps; qd_mm[j] -= eps;

            H[i * cd + j] = (lnn_forward(params, q, qd_pp)
                           - lnn_forward(params, q, qd_pm)
                           - lnn_forward(params, q, qd_mp)
                           + lnn_forward(params, q, qd_mm)) / (4.0f * eps * eps);
        }
    }

    /* Compute (∇_q ∇_{q̇} L) q̇  via finite differences */
    /* For simplicity, use diagonal approximation */
    float *H_qqdot = (float *)alloca(cd * sizeof(float));
    for (size_t i = 0; i < cd; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < cd; j++) {
            sum += H[i * cd + j] * q_dot[j];
        }
        H_qqdot[i] = sum;
    }

    /* Right-hand side: ∂L/∂q - (∇_q ∇_{q̇} L) q̇ */
    float *rhs = (float *)alloca(cd * sizeof(float));
    for (size_t i = 0; i < cd; i++) {
        rhs[i] = dL_dq[i] - H_qqdot[i];
    }

    /* Solve H * q̈ = rhs using simple Gaussian elimination (for small cd) */
    /* Augmented matrix [H | rhs] */
    float *aug = (float *)alloca(cd * (cd + 1) * sizeof(float));
    for (size_t i = 0; i < cd; i++) {
        for (size_t j = 0; j < cd; j++) {
            aug[i * (cd + 1) + j] = H[i * cd + j];
        }
        aug[i * (cd + 1) + cd] = rhs[i];
    }

    /* Forward elimination with partial pivoting */
    for (size_t col = 0; col < cd; col++) {
        /* Find pivot */
        size_t pivot = col;
        float max_val = fabsf(aug[col * (cd + 1) + col]);
        for (size_t row = col + 1; row < cd; row++) {
            float val = fabsf(aug[row * (cd + 1) + col]);
            if (val > max_val) { max_val = val; pivot = row; }
        }
        if (max_val < 1e-10f) {
            /* Singular matrix, set acceleration to 0 */
            memset(q_ddot, 0, cd * sizeof(float));
            return;
        }
        /* Swap rows */
        if (pivot != col) {
            for (size_t j = 0; j <= cd; j++) {
                float tmp = aug[col * (cd + 1) + j];
                aug[col * (cd + 1) + j] = aug[pivot * (cd + 1) + j];
                aug[pivot * (cd + 1) + j] = tmp;
            }
        }
        /* Eliminate */
        for (size_t row = col + 1; row < cd; row++) {
            float factor = aug[row * (cd + 1) + col] / aug[col * (cd + 1) + col];
            for (size_t j = col; j <= cd; j++) {
                aug[row * (cd + 1) + j] -= factor * aug[col * (cd + 1) + j];
            }
        }
    }

    /* Back substitution */
    for (int i = (int)cd - 1; i >= 0; i--) {
        float sum = aug[i * (cd + 1) + cd];
        for (size_t j = i + 1; j < cd; j++) {
            sum -= aug[i * (cd + 1) + j] * q_ddot[j];
        }
        q_ddot[i] = sum / aug[i * (cd + 1) + i];
    }
}
