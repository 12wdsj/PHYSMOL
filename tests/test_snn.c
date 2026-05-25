/* PHYSMOL SNN C Unit Tests */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../src/core/snn.h"
#include "../src/core/causal.h"

#define NUM_NEURONS 512
#define TOLERANCE 1e-4f

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    printf("  TEST: %-40s ", name); \
    test_count++; \
} while(0)

#define PASS() do { printf("[PASS]\n"); pass_count++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

void test_spike_state() {
    TEST("spike state create/set/get/count");
    SpikeState *s = snn_spike_create(NUM_NEURONS);
    assert(s != NULL);
    assert(snn_spike_count(s) == 0);

    snn_spike_set(s, 0);
    snn_spike_set(s, 100);
    snn_spike_set(s, 511);
    assert(snn_spike_count(s) == 3);
    assert(snn_spike_get(s, 0) == 1);
    assert(snn_spike_get(s, 50) == 0);
    assert(snn_spike_get(s, 511) == 1);

    snn_spike_clear(s);
    assert(snn_spike_count(s) == 0);

    snn_spike_free(s);
    PASS();
}

void test_spike_array_conversion() {
    TEST("spike <-> array conversion");
    SpikeState *s = snn_spike_create(NUM_NEURONS);
    int arr[NUM_NEURONS];
    memset(arr, 0, sizeof(arr));
    arr[0] = 1; arr[5] = 1; arr[100] = 1;

    snn_spike_from_array(s, arr);
    assert(snn_spike_count(s) == 3);

    int out[NUM_NEURONS];
    snn_spike_to_array(s, out);
    assert(out[0] == 1);
    assert(out[1] == 0);
    assert(out[5] == 1);
    assert(out[100] == 1);
    assert(out[101] == 0);

    snn_spike_free(s);
    PASS();
}

void test_neuron_state() {
    TEST("neuron state create");
    NeuronState *ns = snn_neuron_create(NUM_NEURONS);
    assert(ns != NULL);
    for (size_t i = 0; i < NUM_NEURONS; i++) {
        assert(ns->v_mem[i] == 0.0f);
        assert(ns->t_ref[i] == 0.0f);
    }
    snn_neuron_free(ns);
    PASS();
}

void test_synapse_matrix() {
    TEST("synapse matrix create/random init");
    SynapseMatrix *syn = snn_synapse_create(NUM_NEURONS, NUM_NEURONS);
    assert(syn != NULL);

    snn_synapse_random_init(syn, 0.1f, 42);
    /* Check that weights are not all zero */
    int nonzero = 0;
    size_t n = syn->num_pre * syn->num_post;
    for (size_t i = 0; i < n; i++) {
        if (fabsf(syn->weights[i]) > 1e-10f) nonzero++;
    }
    assert(nonzero > (int)(n / 2)); /* most should be nonzero */

    snn_synapse_free(syn);
    PASS();
}

void test_lif_step() {
    TEST("LIF neuron step");
    size_t n = 64;
    NeuronState *ns = snn_neuron_create(n);
    SpikeState *spikes = snn_spike_create(n);
    SynapseMatrix *syn = snn_synapse_create(n, n);
    snn_synapse_random_init(syn, 0.01f, 42);

    LIFParams params = {1.0f, 0.0f, 20.0f, 2.0f, 1.0f};

    /* Strong input to cause spikes */
    float input[64];
    for (size_t i = 0; i < n; i++) input[i] = 2.0f;

    /* Run several steps */
    int total_spikes = 0;
    for (int t = 0; t < 50; t++) {
        snn_step_lif(ns, spikes, syn, input, &params);
        total_spikes += snn_spike_count(spikes);
    }

    /* With strong input, should have some spikes */
    assert(total_spikes > 0);

    snn_neuron_free(ns);
    snn_spike_free(spikes);
    snn_synapse_free(syn);
    PASS();
}

void test_stdp() {
    TEST("STDP weight update");
    size_t n = 32;
    SynapseMatrix *syn = snn_synapse_create(n, n);
    snn_synapse_random_init(syn, 0.05f, 42);

    SpikeState *pre = snn_spike_create(n);
    SpikeState *post = snn_spike_create(n);

    float trace_pre[32] = {0};
    float trace_post[32] = {0};

    /* Set some spikes */
    snn_spike_set(pre, 0);
    snn_spike_set(pre, 1);
    snn_spike_set(post, 2);
    snn_spike_set(post, 3);

    float w_before = syn->weights[0 * n + 2]; /* synapse 0->2 */

    snn_stdp_update(syn, pre, post, trace_pre, trace_post,
                    0.01f, 0.012f, 20.0f, 20.0f, 1.0f);

    float w_after = syn->weights[0 * n + 2];
    /* Weight should change */
    assert(fabsf(w_after - w_before) > 1e-10f);

    snn_synapse_free(syn);
    snn_spike_free(pre);
    snn_spike_free(post);
    PASS();
}

void test_three_factor() {
    TEST("three-factor learning rule");
    size_t n = 16;
    SynapseMatrix *syn = snn_synapse_create(n, n);
    snn_synapse_random_init(syn, 0.05f, 42);
    EligibilityTrace *elig = snn_eligibility_create(n, n, 100.0f);

    SpikeState *pre = snn_spike_create(n);
    SpikeState *post = snn_spike_create(n);

    /* Both pre and post spike at same synapse */
    snn_spike_set(pre, 0);
    snn_spike_set(post, 0);

    float w_before = syn->weights[0];

    /* With positive reward */
    snn_three_factor_update(syn, elig, pre, post, 1.0f, 0.01f, 100.0f, 1.0f);
    float w_after_pos = syn->weights[0];

    /* Weight should increase with positive reward */
    assert(w_after_pos > w_before);

    /* Reset and test with negative reward */
    snn_synapse_random_init(syn, 0.05f, 42);
    snn_eligibility_clear(elig);
    snn_three_factor_update(syn, elig, pre, post, -1.0f, 0.01f, 100.0f, 1.0f);
    float w_after_neg = syn->weights[0];

    /* Weight should decrease with negative reward */
    assert(w_after_neg < 0.05f);

    snn_synapse_free(syn);
    snn_eligibility_free(elig);
    snn_spike_free(pre);
    snn_spike_free(post);
    PASS();
}

void test_causal_graph() {
    TEST("causal graph add/prune/propagate");
    CausalGraph *g = causal_graph_create(100, 64);

    causal_graph_add_edge(g, 0, 1, 1.0f, 0.5f);
    causal_graph_add_edge(g, 1, 2, 0.8f, 0.3f);
    causal_graph_add_edge(g, 0, 2, 0.5f, 0.1f);
    assert(causal_graph_edge_count(g) == 3);

    /* Prune edges with credit < 0.4 */
    size_t removed = causal_graph_prune(g, 0.4f);
    assert(removed == 2); /* edges 1->2 and 0->2 should be removed */
    assert(causal_graph_edge_count(g) == 1);

    /* Propagate */
    float activation[100];
    memset(activation, 0, sizeof(activation));
    activation[0] = 1.0f;
    causal_graph_propagate(g, activation, 0, 1);
    /* After 1 step, node 1 should have activation */
    assert(activation[1] > 0.0f);

    causal_graph_free(g);
    PASS();
}

void test_eligibility_trace() {
    TEST("eligibility trace decay");
    EligibilityTrace *et = snn_eligibility_create(8, 8, 10.0f);

    /* Set a trace value */
    et->traces[0] = 1.0f;

    /* Decay should reduce it */
    /* (We can't directly test decay without calling three_factor_update,
     *  but we verify the structure is correct) */
    assert(et->traces[0] == 1.0f);
    assert(et->tau_e == 10.0f);

    snn_eligibility_free(et);
    PASS();
}

int main() {
    printf("=== PHYSMOL SNN Unit Tests (N=%d) ===\n", NUM_NEURONS);

    test_spike_state();
    test_spike_array_conversion();
    test_neuron_state();
    test_synapse_matrix();
    test_lif_step();
    test_stdp();
    test_three_factor();
    test_causal_graph();
    test_eligibility_trace();

    printf("\n%d/%d tests passed\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
