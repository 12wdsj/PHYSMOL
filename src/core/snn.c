#include "snn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Bit manipulation helpers */
#define WORD_IDX(i) ((i) >> 6)         /* i / 64 */
#define BIT_IDX(i)  ((i) & 63)         /* i % 64 */
#define BIT_MASK(i) (1ULL << BIT_IDX(i))

/* === SpikeState === */

SpikeState *snn_spike_create(size_t num_neurons) {
    SpikeState *s = (SpikeState *)calloc(1, sizeof(SpikeState));
    if (!s) return NULL;
    s->num_neurons = num_neurons;
    s->num_words = (num_neurons + 63) / 64;
    s->bits = (uint64_t *)calloc(s->num_words, sizeof(uint64_t));
    if (!s->bits) { free(s); return NULL; }
    return s;
}

void snn_spike_free(SpikeState *state) {
    if (state) {
        free(state->bits);
        free(state);
    }
}

void snn_spike_clear(SpikeState *state) {
    memset(state->bits, 0, state->num_words * sizeof(uint64_t));
}

void snn_spike_set(SpikeState *state, size_t neuron_id) {
    state->bits[WORD_IDX(neuron_id)] |= BIT_MASK(neuron_id);
}

void snn_spike_clear_bit(SpikeState *state, size_t neuron_id) {
    state->bits[WORD_IDX(neuron_id)] &= ~BIT_MASK(neuron_id);
}

int snn_spike_get(const SpikeState *state, size_t neuron_id) {
    return (state->bits[WORD_IDX(neuron_id)] & BIT_MASK(neuron_id)) ? 1 : 0;
}

size_t snn_spike_count(const SpikeState *state) {
    size_t count = 0;
    for (size_t i = 0; i < state->num_words; i++) {
        /* Popcount */
        uint64_t w = state->bits[i];
        /* Brian Kernighan's method */
        while (w) { count++; w &= w - 1; }
    }
    return count;
}

void snn_spike_to_array(const SpikeState *state, int *out) {
    for (size_t i = 0; i < state->num_neurons; i++) {
        out[i] = snn_spike_get(state, i);
    }
}

void snn_spike_from_array(SpikeState *state, const int *in) {
    snn_spike_clear(state);
    for (size_t i = 0; i < state->num_neurons; i++) {
        if (in[i]) snn_spike_set(state, i);
    }
}

/* === NeuronState === */

NeuronState *snn_neuron_create(size_t num_neurons) {
    NeuronState *ns = (NeuronState *)calloc(1, sizeof(NeuronState));
    if (!ns) return NULL;
    ns->num_neurons = num_neurons;
    ns->v_mem = (float *)calloc(num_neurons, sizeof(float));
    ns->t_ref = (float *)calloc(num_neurons, sizeof(float));
    if (!ns->v_mem || !ns->t_ref) {
        free(ns->v_mem); free(ns->t_ref); free(ns);
        return NULL;
    }
    return ns;
}

void snn_neuron_free(NeuronState *state) {
    if (state) {
        free(state->v_mem);
        free(state->t_ref);
        free(state);
    }
}

/* === SynapseMatrix === */

SynapseMatrix *snn_synapse_create(size_t num_pre, size_t num_post) {
    SynapseMatrix *syn = (SynapseMatrix *)calloc(1, sizeof(SynapseMatrix));
    if (!syn) return NULL;
    syn->num_pre = num_pre;
    syn->num_post = num_post;
    syn->weights = (float *)calloc(num_pre * num_post, sizeof(float));
    if (!syn->weights) { free(syn); return NULL; }
    return syn;
}

void snn_synapse_free(SynapseMatrix *syn) {
    if (syn) {
        free(syn->weights);
        free(syn);
    }
}

/* Simple xorshift64 for weight init */
static uint64_t _snn_xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

void snn_synapse_random_init(SynapseMatrix *syn, float scale, uint64_t seed) {
    uint64_t rng = seed ? seed : 12345;
    size_t n = syn->num_pre * syn->num_post;
    for (size_t i = 0; i < n; i++) {
        /* Box-Muller for Gaussian */
        uint64_t r1 = _snn_xorshift64(&rng);
        uint64_t r2 = _snn_xorshift64(&rng);
        double u1 = (double)(r1 & 0xFFFFFFFF) / (double)0xFFFFFFFF;
        double u2 = (double)(r2 & 0xFFFFFFFF) / (double)0xFFFFFFFF;
        if (u1 < 1e-10) u1 = 1e-10;
        double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
        syn->weights[i] = (float)(z * scale);
    }
}

/* === EligibilityTrace === */

EligibilityTrace *snn_eligibility_create(size_t num_pre, size_t num_post, float tau_e) {
    EligibilityTrace *et = (EligibilityTrace *)calloc(1, sizeof(EligibilityTrace));
    if (!et) return NULL;
    et->num_pre = num_pre;
    et->num_post = num_post;
    et->tau_e = tau_e;
    et->traces = (float *)calloc(num_pre * num_post, sizeof(float));
    if (!et->traces) { free(et); return NULL; }
    return et;
}

void snn_eligibility_free(EligibilityTrace *trace) {
    if (trace) {
        free(trace->traces);
        free(trace);
    }
}

void snn_eligibility_clear(EligibilityTrace *trace) {
    memset(trace->traces, 0, trace->num_pre * trace->num_post * sizeof(float));
}

/* === Core simulation === */

void snn_step_lif(NeuronState *neurons, SpikeState *spikes,
                  const SynapseMatrix *weights,
                  const float *input_current,
                  const LIFParams *params) {
    size_t n = neurons->num_neurons;
    float dt = params->dt;
    float tau_mem = params->tau_mem;
    float v_thresh = params->v_threshold;
    float v_reset = params->v_reset;
    float tau_ref = params->tau_ref;

    /* Decay factor for membrane potential */
    float decay = expf(-dt / tau_mem);

    /* Clear output spikes */
    snn_spike_clear(spikes);

    for (size_t i = 0; i < n; i++) {
        /* Check refractory period */
        if (neurons->t_ref[i] > 0.0f) {
            neurons->t_ref[i] -= dt;
            continue;
        }

        /* Synaptic input: sum of weighted pre-synaptic spikes */
        float I_syn = 0.0f;
        if (weights && spikes) {
            /* For each pre-synaptic neuron that spiked, add its weight */
            /* Note: this is a simplified dense multiplication.
             * For efficiency with sparse spikes, a CSR matrix would be better.
             */
            size_t num_pre = weights->num_pre;
            for (size_t j = 0; j < num_pre; j++) {
                if (snn_spike_get(spikes, j)) {
                    I_syn += weights->weights[j * n + i];
                }
            }
        }

        /* LIF dynamics: tau_mem * dv/dt = -(v - v_rest) + R*I */
        /* v_rest = 0 for simplicity */
        float I_total = I_syn + (input_current ? input_current[i] : 0.0f);
        neurons->v_mem[i] = neurons->v_mem[i] * decay + I_total * dt;

        /* Spike check */
        if (neurons->v_mem[i] >= v_thresh) {
            snn_spike_set(spikes, i);
            neurons->v_mem[i] = v_reset;
            neurons->t_ref[i] = tau_ref;
        }
    }
}

void snn_stdp_update(SynapseMatrix *weights,
                     const SpikeState *pre_spikes,
                     const SpikeState *post_spikes,
                     float *trace_pre,
                     float *trace_post,
                     float a_plus,
                     float a_minus,
                     float tau_plus,
                     float tau_minus,
                     float dt) {
    size_t num_pre = weights->num_pre;
    size_t num_post = weights->num_post;

    /* Decay factors */
    float decay_pre = expf(-dt / tau_plus);
    float decay_post = expf(-dt / tau_minus);

    /* Update traces and apply STDP */
    for (size_t i = 0; i < num_pre; i++) {
        /* Update pre-synaptic trace */
        trace_pre[i] *= decay_pre;
        if (snn_spike_get(pre_spikes, i)) {
            trace_pre[i] += 1.0f;
        }
    }

    for (size_t j = 0; j < num_post; j++) {
        /* Update post-synaptic trace */
        trace_post[j] *= decay_post;
        if (snn_spike_get(post_spikes, j)) {
            trace_post[j] += 1.0f;
        }
    }

    /* Weight update: for each synapse (i->j):
     * If pre fires: dw -= a_minus * post_trace[j]  (LTD)
     * If post fires: dw += a_plus * pre_trace[i]   (LTP)
     */
    for (size_t i = 0; i < num_pre; i++) {
        if (snn_spike_get(pre_spikes, i)) {
            for (size_t j = 0; j < num_post; j++) {
                weights->weights[i * num_post + j] -= a_minus * trace_post[j];
            }
        }
    }

    for (size_t j = 0; j < num_post; j++) {
        if (snn_spike_get(post_spikes, j)) {
            for (size_t i = 0; i < num_pre; i++) {
                weights->weights[i * num_post + j] += a_plus * trace_pre[i];
            }
        }
    }
}

void snn_three_factor_update(SynapseMatrix *weights,
                             EligibilityTrace *elig,
                             const SpikeState *pre_spikes,
                             const SpikeState *post_spikes,
                             float reward,
                             float eta,
                             float tau_e,
                             float dt) {
    size_t num_pre = weights->num_pre;
    size_t num_post = weights->num_post;

    float decay_e = expf(-dt / tau_e);

    /* Update eligibility trace: E_ij += STDP(Delta_t), then decay */
    for (size_t i = 0; i < num_pre; i++) {
        for (size_t j = 0; j < num_post; j++) {
            size_t idx = i * num_post + j;

            /* Decay existing trace */
            elig->traces[idx] *= decay_e;

            /* Add new STDP contribution: pre * post co-activation */
            if (snn_spike_get(pre_spikes, i) && snn_spike_get(post_spikes, j)) {
                elig->traces[idx] += 1.0f;
            }

            /* Three-factor weight update: dw = R(t) * eta * E_ij */
            weights->weights[idx] += reward * eta * elig->traces[idx];
        }
    }
}
