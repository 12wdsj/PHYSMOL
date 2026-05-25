#include "causal.h"
#include <stdlib.h>
#include <string.h>

CausalGraph *causal_graph_create(size_t max_nodes, size_t initial_capacity) {
    CausalGraph *g = (CausalGraph *)calloc(1, sizeof(CausalGraph));
    if (!g) return NULL;
    g->max_nodes = max_nodes;
    g->capacity = initial_capacity ? initial_capacity : 1024;
    g->num_edges = 0;
    g->edges = (CausalEdge *)calloc(g->capacity, sizeof(CausalEdge));
    if (!g->edges) { free(g); return NULL; }
    return g;
}

void causal_graph_free(CausalGraph *graph) {
    if (graph) {
        free(graph->edges);
        free(graph);
    }
}

/* Find edge by (pre, post), return index or -1 */
int causal_graph_find_edge(const CausalGraph *graph, size_t pre, size_t post) {
    for (size_t i = 0; i < graph->num_edges; i++) {
        if (graph->edges[i].pre == pre && graph->edges[i].post == post) {
            return (int)i;
        }
    }
    return -1;
}

size_t causal_graph_add_edge(CausalGraph *graph, size_t pre, size_t post,
                             float weight, float credit) {
    /* Check if edge exists */
    int idx = causal_graph_find_edge(graph, pre, post);
    if (idx >= 0) {
        /* Update existing edge */
        graph->edges[idx].weight = weight;
        graph->edges[idx].credit = credit;
        return (size_t)idx;
    }

    /* Grow if needed */
    if (graph->num_edges >= graph->capacity) {
        size_t new_cap = graph->capacity * 2;
        CausalEdge *new_edges = (CausalEdge *)realloc(
            graph->edges, new_cap * sizeof(CausalEdge));
        if (!new_edges) return (size_t)-1;
        graph->edges = new_edges;
        graph->capacity = new_cap;
    }

    size_t idx_new = graph->num_edges;
    graph->edges[idx_new].pre = pre;
    graph->edges[idx_new].post = post;
    graph->edges[idx_new].weight = weight;
    graph->edges[idx_new].credit = credit;
    graph->num_edges++;

    /* Update max_nodes if needed */
    size_t max_n = (pre > post ? pre : post) + 1;
    if (max_n > graph->max_nodes) graph->max_nodes = max_n;

    return idx_new;
}

size_t causal_graph_prune(CausalGraph *graph, float threshold) {
    size_t removed = 0;
    size_t write = 0;
    for (size_t i = 0; i < graph->num_edges; i++) {
        if (graph->edges[i].credit >= threshold) {
            if (write != i) {
                graph->edges[write] = graph->edges[i];
            }
            write++;
        } else {
            removed++;
        }
    }
    graph->num_edges = write;
    return removed;
}

void causal_graph_reinforce(CausalGraph *graph,
                            const int *active_nodes, size_t num_active,
                            float reward) {
    for (size_t i = 0; i < graph->num_edges; i++) {
        int pre_active = 0, post_active = 0;
        for (size_t j = 0; j < num_active; j++) {
            if ((size_t)active_nodes[j] == graph->edges[i].pre) pre_active = 1;
            if ((size_t)active_nodes[j] == graph->edges[i].post) post_active = 1;
            if (pre_active && post_active) break;
        }
        if (pre_active && post_active) {
            graph->edges[i].credit += reward;
        }
    }
}

size_t causal_graph_outgoing(const CausalGraph *graph, size_t node,
                             CausalEdge *out_edges, size_t max_out) {
    size_t count = 0;
    for (size_t i = 0; i < graph->num_edges && count < max_out; i++) {
        if (graph->edges[i].pre == node) {
            out_edges[count++] = graph->edges[i];
        }
    }
    return count;
}

size_t causal_graph_incoming(const CausalGraph *graph, size_t node,
                             CausalEdge *out_edges, size_t max_out) {
    size_t count = 0;
    for (size_t i = 0; i < graph->num_edges && count < max_out; i++) {
        if (graph->edges[i].post == node) {
            out_edges[count++] = graph->edges[i];
        }
    }
    return count;
}

size_t causal_graph_edge_count(const CausalGraph *graph) {
    return graph->num_edges;
}

void causal_graph_propagate(const CausalGraph *graph,
                            float *activation,
                            size_t source,
                            int steps) {
    /* Allocate temp buffer for next step */
    float *next = (float *)calloc(graph->max_nodes, sizeof(float));
    if (!next) return;

    for (int step = 0; step < steps; step++) {
        memset(next, 0, graph->max_nodes * sizeof(float));

        /* For each edge, propagate activation */
        for (size_t i = 0; i < graph->num_edges; i++) {
            size_t pre = graph->edges[i].pre;
            size_t post = graph->edges[i].post;
            next[post] += activation[pre] * graph->edges[i].weight;
        }

        /* Apply ReLU-like activation */
        for (size_t j = 0; j < graph->max_nodes; j++) {
            if (next[j] < 0.0f) next[j] = 0.0f;
            if (next[j] > 1.0f) next[j] = 1.0f;
        }

        /* Swap */
        memcpy(activation, next, graph->max_nodes * sizeof(float));
    }

    free(next);
}
