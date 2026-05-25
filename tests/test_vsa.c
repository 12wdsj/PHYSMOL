/* PHYSMOL VSA C Unit Tests */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../src/core/vsa.h"

#define DIM 4096
#define TOLERANCE 1e-4f

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    printf("  TEST: %-40s ", name); \
    test_count++; \
} while(0)

#define PASS() do { printf("[PASS]\n"); pass_count++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); } while(0)

void test_create_free() {
    TEST("create and free vector");
    VSAVector *v = vsa_create(DIM);
    assert(v != NULL);
    assert(v->dim == DIM);
    for (size_t i = 0; i < DIM; i++) assert(v->data[i] == 0.0f);
    vsa_free(v);
    PASS();
}

void test_random_bipolar() {
    TEST("random bipolar generation");
    VSAVector *v = vsa_create(DIM);
    vsa_random_bipolar(v, 42);
    int has_pos = 0, has_neg = 0;
    for (size_t i = 0; i < DIM; i++) {
        if (v->data[i] == 1.0f) has_pos = 1;
        else if (v->data[i] == -1.0f) has_neg = 1;
        else { FAIL("non-bipolar value found"); return; }
    }
    assert(has_pos && has_neg);
    vsa_free(v);
    PASS();
}

void test_bind() {
    TEST("binding (Hadamard product)");
    VSAVector *a = vsa_create(DIM);
    VSAVector *b = vsa_create(DIM);
    VSAVector *out = vsa_create(DIM);
    vsa_random_bipolar(a, 1);
    vsa_random_bipolar(b, 2);
    vsa_bind(out, a, b);
    for (size_t i = 0; i < DIM; i++) {
        float expected = a->data[i] * b->data[i];
        assert(fabsf(out->data[i] - expected) < TOLERANCE);
    }
    vsa_free(a); vsa_free(b); vsa_free(out);
    PASS();
}

void test_bind_commutativity() {
    TEST("binding commutativity: a*b == b*a");
    VSAVector *a = vsa_create(DIM);
    VSAVector *b = vsa_create(DIM);
    VSAVector *ab = vsa_create(DIM);
    VSAVector *ba = vsa_create(DIM);
    vsa_random_bipolar(a, 10);
    vsa_random_bipolar(b, 20);
    vsa_bind(ab, a, b);
    vsa_bind(ba, b, a);
    for (size_t i = 0; i < DIM; i++) {
        assert(fabsf(ab->data[i] - ba->data[i]) < TOLERANCE);
    }
    vsa_free(a); vsa_free(b); vsa_free(ab); vsa_free(ba);
    PASS();
}

void test_bundle() {
    TEST("bundling (addition)");
    VSAVector *a = vsa_create(DIM);
    VSAVector *b = vsa_create(DIM);
    VSAVector *out = vsa_create(DIM);
    vsa_random_bipolar(a, 3);
    vsa_random_bipolar(b, 4);
    vsa_bundle(out, a, b);
    for (size_t i = 0; i < DIM; i++) {
        float expected = a->data[i] + b->data[i];
        assert(fabsf(out->data[i] - expected) < TOLERANCE);
    }
    vsa_free(a); vsa_free(b); vsa_free(out);
    PASS();
}

void test_cosine_similarity() {
    TEST("cosine similarity");
    VSAVector *a = vsa_create(DIM);
    VSAVector *b = vsa_create(DIM);
    vsa_random_bipolar(a, 5);
    /* Self-similarity should be 1.0 */
    float sim_aa = vsa_cosine_similarity(a, a);
    assert(fabsf(sim_aa - 1.0f) < TOLERANCE);
    /* Random vectors should be nearly orthogonal */
    vsa_random_bipolar(b, 6);
    float sim_ab = vsa_cosine_similarity(a, b);
    assert(fabsf(sim_ab) < 0.1f); /* within 0.1 with high probability */
    vsa_free(a); vsa_free(b);
    PASS();
}

void test_quasi_orthogonality() {
    TEST("quasi-orthogonality of random vectors");
    int num_vecs = 100;
    VSAVector **vecs = (VSAVector **)malloc(num_vecs * sizeof(VSAVector *));
    for (int i = 0; i < num_vecs; i++) {
        vecs[i] = vsa_create(DIM);
        vsa_random_bipolar(vecs[i], i + 100);
    }
    float max_sim = 0.0f;
    int count_above_threshold = 0;
    for (int i = 0; i < num_vecs; i++) {
        for (int j = i + 1; j < num_vecs; j++) {
            float sim = fabsf(vsa_cosine_similarity(vecs[i], vecs[j]));
            if (sim > max_sim) max_sim = sim;
            if (sim > 0.1f) count_above_threshold++;
        }
    }
    /* With D=4096, σ=1/√4096≈0.016, so |S|>0.1 should be rare */
    printf("(max_sim=%.4f, pairs_above_0.1=%d) ", max_sim, count_above_threshold);
    assert(max_sim < 0.15f);
    for (int i = 0; i < num_vecs; i++) vsa_free(vecs[i]);
    free(vecs);
    PASS();
}

void test_permute() {
    TEST("circular permutation");
    VSAVector *v = vsa_create(DIM);
    VSAVector *out = vsa_create(DIM);
    vsa_random_bipolar(v, 7);
    vsa_permute(out, v, 1);
    /* Shifted by 1: out[0] == v[1], out[DIM-1] == v[0] */
    assert(fabsf(out->data[0] - v->data[1]) < TOLERANCE);
    assert(fabsf(out->data[DIM - 1] - v->data[0]) < TOLERANCE);
    vsa_free(v); vsa_free(out);
    PASS();
}

void test_normalize() {
    TEST("normalize to unit length");
    VSAVector *v = vsa_create(DIM);
    vsa_random_bipolar(v, 8);
    vsa_normalize(v);
    float norm_sq = 0.0f;
    for (size_t i = 0; i < DIM; i++) norm_sq += v->data[i] * v->data[i];
    assert(fabsf(norm_sq - 1.0f) < TOLERANCE);
    vsa_free(v);
    PASS();
}

void test_codebook() {
    TEST("codebook add/lookup/nearest");
    VSACodebook *cb = vsa_codebook_create(DIM, 16);
    VSAVector *red = vsa_create(DIM);
    VSAVector *blue = vsa_create(DIM);
    VSAVector *green = vsa_create(DIM);
    vsa_random_bipolar(red, 100);
    vsa_random_bipolar(blue, 200);
    vsa_random_bipolar(green, 300);

    vsa_codebook_add(cb, "red", red);
    vsa_codebook_add(cb, "blue", blue);
    vsa_codebook_add(cb, "green", green);
    assert(cb->num_vectors == 3);

    /* Lookup */
    const VSAVector *found = vsa_codebook_lookup(cb, "red");
    assert(found != NULL);

    /* Nearest: query with red, should find red */
    float sim;
    size_t idx = vsa_codebook_nearest(cb, red, &sim);
    assert(idx == 0); /* red was added first */
    assert(sim > 0.99f);

    vsa_free(red); vsa_free(blue); vsa_free(green);
    vsa_codebook_free(cb);
    PASS();
}

void test_fpe_encode() {
    TEST("FPE spatial encoding");
    VSAVector *out = vsa_create(DIM);
    VSAVector *bx = vsa_create(DIM);
    VSAVector *by = vsa_create(DIM);
    VSAVector *bz = vsa_create(DIM);
    vsa_random_phase(bx, 500);
    vsa_random_phase(by, 600);
    vsa_random_phase(bz, 700);

    vsa_fpe_encode(out, 1.0f, 2.0f, 3.0f, bx, by, bz);

    /* Verify: phase should be x*bx + y*by + z*bz */
    for (size_t i = 0; i < DIM; i++) {
        float expected = 1.0f * bx->data[i] + 2.0f * by->data[i] + 3.0f * bz->data[i];
        assert(fabsf(out->data[i] - expected) < TOLERANCE);
    }

    vsa_free(out); vsa_free(bx); vsa_free(by); vsa_free(bz);
    PASS();
}

int main() {
    printf("=== PHYSMOL VSA Unit Tests (DIM=%d) ===\n", DIM);

    test_create_free();
    test_random_bipolar();
    test_bind();
    test_bind_commutativity();
    test_bundle();
    test_cosine_similarity();
    test_quasi_orthogonality();
    test_permute();
    test_normalize();
    test_codebook();
    test_fpe_encode();

    printf("\n%d/%d tests passed\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
