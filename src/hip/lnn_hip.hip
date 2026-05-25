/*
 * PHYSMOL HIP/ROCm GPU Kernels
 *
 * This file contains GPU-accelerated implementations for:
 * 1. LNN Lagrangian forward pass
 * 2. VSA batch binding operations
 * 3. Mass matrix (Hessian) computation
 *
 * Target: AMD MI250 (gfx90a) or MI300 (gfx942)
 * Build: hipcc -O3 --offload-arch=gfx90a lnn_hip.hip -c
 *
 * NOTE: This is a template. The actual kernels need to be tuned
 * for the specific AMD GPU architecture on the cloud server.
 * On the local development machine (no AMD GPU), the CPU fallback
 * in lnn.c is used automatically.
 */

#include "lnn_hip.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>

/* === Error checking === */
#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* === MLP Forward Kernel === */
/* Each thread computes one output neuron */
__global__ void mlp_layer_kernel(const float *input, const float *weights,
                                  const float *bias, float *output,
                                  int in_dim, int out_dim, int use_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim) return;

    float sum = bias[idx];
    for (int i = 0; i < in_dim; i++) {
        sum += input[i] * weights[i * out_dim + idx];
    }
    output[idx] = use_relu ? fmaxf(0.0f, sum) : sum;
}

__global__ void concat_state_kernel(const float *q, const float *q_dot,
                                    float *input,
                                    int batch_size, int coord_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * coord_dim;
    if (idx >= total) return;

    int batch = idx / coord_dim;
    int dim = idx % coord_dim;
    int input_dim = coord_dim * 2;
    input[batch * input_dim + dim] = q[idx];
    input[batch * input_dim + coord_dim + dim] = q_dot[idx];
}

__global__ void mlp_layer_batch_kernel(const float *input,
                                       const float *weights,
                                       const float *bias,
                                       float *output,
                                       int batch_size,
                                       int in_dim,
                                       int out_dim,
                                       int use_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_dim;
    if (idx >= total) return;

    int batch = idx / out_dim;
    int out = idx % out_dim;

    float sum = bias ? bias[out] : 0.0f;
    const float *x = input + batch * in_dim;
    for (int i = 0; i < in_dim; i++) {
        sum += x[i] * weights[i * out_dim + out];
    }
    output[idx] = use_relu ? fmaxf(0.0f, sum) : sum;
}

/* === VSA Batch Bind Kernel === */
__global__ void vsa_bind_kernel(float *out, const float *a, const float *b,
                                 int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = a[idx] * b[idx];
}

/* === Cosine Similarity Kernel === */
__global__ void cosine_similarity_kernel(const float *a, const float *b,
                                         float *sim_out, int dim) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pair = blockIdx.x;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    int base = pair * dim;
    for (int i = tid; i < dim; i += blockDim.x) {
        float av = a[base + i];
        float bv = b[base + i];
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    sdata[tid] = dot;
    sdata[blockDim.x + tid] = norm_a;
    sdata[2 * blockDim.x + tid] = norm_b;
    __syncthreads();

    /* Parallel reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
            sdata[2 * blockDim.x + tid] += sdata[2 * blockDim.x + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = sqrtf(sdata[blockDim.x] * sdata[2 * blockDim.x]) + 1e-12f;
        sim_out[pair] = sdata[0] / denom;
    }
}

/* === Host-callable functions === */

int lnn_hip_available(void) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return (err == hipSuccess && device_count > 0) ? 1 : 0;
}

int lnn_forward_hip(const float *q, const float *q_dot,
                    const float *params,
                    float *L_out,
                    size_t batch_size, size_t coord_dim, size_t hidden_dim) {
    /* Flattened params layout must match src/core/lnn.c:
     *   w1[input_dim * hidden_dim], b1[hidden_dim],
     *   w2[hidden_dim * hidden_dim], b2[hidden_dim],
     *   w3[hidden_dim], b3[1]
     */

    size_t input_dim = 2 * coord_dim;
    size_t w1_size = input_dim * hidden_dim;
    size_t b1_size = hidden_dim;
    size_t w2_size = hidden_dim * hidden_dim;
    size_t b2_size = hidden_dim;
    size_t w3_size = hidden_dim;
    size_t param_count = w1_size + b1_size + w2_size + b2_size + w3_size + 1;

    /* Allocate device memory */
    float *d_q, *d_q_dot, *d_params, *d_L_out;
    float *d_h1, *d_h2, *d_input;

    HIP_CHECK(hipMalloc(&d_q, batch_size * coord_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q_dot, batch_size * coord_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_input, batch_size * input_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_h1, batch_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_h2, batch_size * hidden_dim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_L_out, batch_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_params, param_count * sizeof(float)));

    /* Copy input to device */
    HIP_CHECK(hipMemcpy(d_q, q, batch_size * coord_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q_dot, q_dot, batch_size * coord_dim * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_params, params, param_count * sizeof(float),
                        hipMemcpyHostToDevice));

    const float *w1 = d_params;
    const float *b1 = w1 + w1_size;
    const float *w2 = b1 + b1_size;
    const float *b2 = w2 + w2_size;
    const float *w3 = b2 + b2_size;
    const float *b3 = w3 + w3_size;

    int block_size = 256;
    int state_total = (int)(batch_size * coord_dim);
    int state_grid = (state_total + block_size - 1) / block_size;
    hipLaunchKernelGGL(concat_state_kernel, dim3(state_grid), dim3(block_size),
                       0, 0, d_q, d_q_dot, d_input,
                       (int)batch_size, (int)coord_dim);

    int h_total = (int)(batch_size * hidden_dim);
    int h_grid = (h_total + block_size - 1) / block_size;
    hipLaunchKernelGGL(mlp_layer_batch_kernel, dim3(h_grid), dim3(block_size),
                       0, 0, d_input, w1, b1, d_h1,
                       (int)batch_size, (int)input_dim, (int)hidden_dim, 1);
    hipLaunchKernelGGL(mlp_layer_batch_kernel, dim3(h_grid), dim3(block_size),
                       0, 0, d_h1, w2, b2, d_h2,
                       (int)batch_size, (int)hidden_dim, (int)hidden_dim, 1);

    int out_grid = ((int)batch_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(mlp_layer_batch_kernel, dim3(out_grid), dim3(block_size),
                       0, 0, d_h2, w3, b3, d_L_out,
                       (int)batch_size, (int)hidden_dim, 1, 0);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(L_out, d_L_out, batch_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    /* Cleanup */
    hipFree(d_q); hipFree(d_q_dot); hipFree(d_input);
    hipFree(d_h1); hipFree(d_h2); hipFree(d_L_out); hipFree(d_params);

    return 0;
}

int lnn_mass_matrix_hip(const float *q, const float *q_dot,
                        const float *params, float *M_out,
                        size_t coord_dim, size_t hidden_dim) {
    (void)q; (void)q_dot; (void)params; (void)M_out;
    (void)coord_dim; (void)hidden_dim;
    fprintf(stderr, "lnn_mass_matrix_hip: use PyTorch autograd path for Hessian on ROCm\n");
    return -1;
}

int vsa_bind_hip(float *out, const float *a, const float *b,
                 size_t num_vectors, size_t dim) {
    size_t total = num_vectors * dim;

    float *d_out, *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_out, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_a, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, total * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_a, a, total * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b, total * sizeof(float), hipMemcpyHostToDevice));

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    hipLaunchKernelGGL(vsa_bind_kernel, dim3(grid_size), dim3(block_size),
                       0, 0, d_out, d_a, d_b, (int)total);

    HIP_CHECK(hipMemcpy(out, d_out, total * sizeof(float), hipMemcpyDeviceToHost));

    hipFree(d_out); hipFree(d_a); hipFree(d_b);
    return 0;
}

int vsa_similarity_batch_hip(const float *a, const float *b,
                             float *sim_out,
                             size_t num_pairs, size_t dim) {
    size_t total = num_pairs * dim;

    float *d_a, *d_b, *d_sim;
    HIP_CHECK(hipMalloc(&d_a, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sim, num_pairs * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_a, a, total * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b, total * sizeof(float), hipMemcpyHostToDevice));

    int block_size = 256;
    size_t shared = 3 * block_size * sizeof(float);
    hipLaunchKernelGGL(cosine_similarity_kernel, dim3((int)num_pairs), dim3(block_size),
                       shared, 0, d_a, d_b, d_sim, (int)dim);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(sim_out, d_sim, num_pairs * sizeof(float),
                        hipMemcpyDeviceToHost));

    hipFree(d_a); hipFree(d_b); hipFree(d_sim);
    return 0;
}

#else /* No HIP available */

int lnn_hip_available(void) { return 0; }

int lnn_forward_hip(const float *q, const float *q_dot,
                    const float *params, float *L_out,
                    size_t batch_size, size_t coord_dim, size_t hidden_dim) {
    (void)q; (void)q_dot; (void)params; (void)L_out;
    (void)batch_size; (void)coord_dim; (void)hidden_dim;
    return -1; /* Not available */
}

int lnn_mass_matrix_hip(const float *q, const float *q_dot,
                        const float *params, float *M_out,
                        size_t coord_dim, size_t hidden_dim) {
    (void)q; (void)q_dot; (void)params; (void)M_out;
    (void)coord_dim; (void)hidden_dim;
    return -1;
}

int vsa_bind_hip(float *out, const float *a, const float *b,
                 size_t num_vectors, size_t dim) {
    (void)out; (void)a; (void)b; (void)num_vectors; (void)dim;
    return -1;
}

int vsa_similarity_batch_hip(const float *a, const float *b,
                             float *sim_out,
                             size_t num_pairs, size_t dim) {
    (void)a; (void)b; (void)sim_out; (void)num_pairs; (void)dim;
    return -1;
}

#endif /* __HIP_PLATFORM_AMD__ */
