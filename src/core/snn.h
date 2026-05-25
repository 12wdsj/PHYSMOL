#ifndef PHYSMOL_SNN_H
#define PHYSMOL_SNN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Spike State (bit-compressed) === */

/* Each neuron's spike is 1 bit. Packed into uint64_t words.
 * 4096 neurons = 64 words = 512 bytes
 */
typedef struct {
    uint64_t *bits;       /* bit-packed spike state */
    size_t num_neurons;
    size_t num_words;     /* ceil(num_neurons / 64) */
} SpikeState;

/* === LIF Neuron Parameters === */
typedef struct {
    float v_threshold;    /* spike threshold (default: 1.0) */
    float v_reset;        /* reset potential (default: 0.0) */
    float tau_mem;        /* membrane time constant (default: 20.0 ms) */
    float tau_ref;        /* refractory period (default: 2.0 ms) */
    float dt;             /* simulation timestep (default: 1.0 ms) */
} LIFParams;

/* === Synapse weight matrix (dense, float32) === */
typedef struct {
    float *weights;       /* num_pre * num_post */
    size_t num_pre;
    size_t num_post;
} SynapseMatrix;

/* === Eligibility traces for three-factor learning === */
typedef struct {
    float *traces;        /* num_pre * num_post */
    size_t num_pre;
    size_t num_post;
    float tau_e;          /* trace decay constant */
} EligibilityTrace;

/* === Neuron state (membrane potentials, refractory counters) === */
typedef struct {
    float *v_mem;         /* membrane potentials */
    float *t_ref;         /* refractory time counters */
    size_t num_neurons;
} NeuronState;

/* === SpikeState lifecycle === */
SpikeState *snn_spike_create(size_t num_neurons);
void snn_spike_free(SpikeState *state);
void snn_spike_clear(SpikeState *state);

/* Set/clear/get individual spike bits */
void snn_spike_set(SpikeState *state, size_t neuron_id);
void snn_spike_clear_bit(SpikeState *state, size_t neuron_id);
int  snn_spike_get(const SpikeState *state, size_t neuron_id);

/* Get spike count */
size_t snn_spike_count(const SpikeState *state);

/* Copy spikes to int array (0/1) for Python */
void snn_spike_to_array(const SpikeState *state, int *out);

/* Set spikes from int array (0/1) */
void snn_spike_from_array(SpikeState *state, const int *in);

/* === NeuronState lifecycle === */
NeuronState *snn_neuron_create(size_t num_neurons);
void snn_neuron_free(NeuronState *state);

/* === SynapseMatrix lifecycle === */
SynapseMatrix *snn_synapse_create(size_t num_pre, size_t num_post);
void snn_synapse_free(SynapseMatrix *syn);
void snn_synapse_random_init(SynapseMatrix *syn, float scale, uint64_t seed);

/* === EligibilityTrace lifecycle === */
EligibilityTrace *snn_eligibility_create(size_t num_pre, size_t num_post, float tau_e);
void snn_eligibility_free(EligibilityTrace *trace);
void snn_eligibility_clear(EligibilityTrace *trace);

/* === Core simulation === */

/* Single LIF step: update neuron states, produce spikes
 * Modifies neuron_state and spike_state in-place.
 */
void snn_step_lif(NeuronState *neurons, SpikeState *spikes,
                  const SynapseMatrix *weights,
                  const float *input_current,
                  const LIFParams *params);

/* STDP update: modify weights based on pre/post spike timing
 * Uses trace-based STDP: each neuron maintains a trace that decays exponentially.
 */
void snn_stdp_update(SynapseMatrix *weights,
                     const SpikeState *pre_spikes,
                     const SpikeState *post_spikes,
                     float *trace_pre,    /* dim: num_pre */
                     float *trace_post,   /* dim: num_post */
                     float a_plus,        /* LTP learning rate */
                     float a_minus,       /* LTD learning rate */
                     float tau_plus,      /* pre-synaptic trace decay */
                     float tau_minus,     /* post-synaptic trace decay */
                     float dt);

/* Three-factor learning rule (Eq.18-20):
 * Delta_w = M(R, Error) * eta * STDP(Delta_t)
 * where M is the global modulation signal (reward/salience)
 */
void snn_three_factor_update(SynapseMatrix *weights,
                             EligibilityTrace *elig,
                             const SpikeState *pre_spikes,
                             const SpikeState *post_spikes,
                             float reward,          /* global reward signal R(t) */
                             float eta,             /* learning rate */
                             float tau_e,           /* eligibility trace decay */
                             float dt);

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_SNN_H */
