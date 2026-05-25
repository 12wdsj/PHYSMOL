#ifndef PHYSMOL_CAUSAL_H
#define PHYSMOL_CAUSAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Causal edge: directed edge from pre_node -> post_node */
typedef struct {
    size_t pre;
    size_t post;
    float weight;     /* causal strength */
    float credit;     /* accumulated credit score */
} CausalEdge;

/* Causal graph: adjacency list representation */
typedef struct {
    CausalEdge *edges;
    size_t num_edges;
    size_t capacity;
    size_t max_nodes;  /* maximum node ID + 1 */
} CausalGraph;

/* === Lifecycle === */
CausalGraph *causal_graph_create(size_t max_nodes, size_t initial_capacity);
void causal_graph_free(CausalGraph *graph);

/* === Edge operations === */

/* Add or update an edge. If edge (pre, post) exists, update weight/credit.
 * Otherwise, add new edge. Returns edge index. */
size_t causal_graph_add_edge(CausalGraph *graph, size_t pre, size_t post,
                             float weight, float credit);

/* Find edge (pre, post). Returns -1 if not found. */
int causal_graph_find_edge(const CausalGraph *graph, size_t pre, size_t post);

/* Remove edges with credit below threshold. Returns number removed. */
size_t causal_graph_prune(CausalGraph *graph, float threshold);

/* Increment credit for all edges that involve the given nodes */
void causal_graph_reinforce(CausalGraph *graph,
                            const int *active_nodes, size_t num_active,
                            float reward);

/* === Query === */

/* Get all outgoing edges from a node. Returns count, fills out_edges. */
size_t causal_graph_outgoing(const CausalGraph *graph, size_t node,
                             CausalEdge *out_edges, size_t max_out);

/* Get all incoming edges to a node. Returns count, fills out_edges. */
size_t causal_graph_incoming(const CausalGraph *graph, size_t node,
                             CausalEdge *out_edges, size_t max_out);

/* Get total number of edges */
size_t causal_graph_edge_count(const CausalGraph *graph);

/* === Counterfactual reasoning === */

/* Propagate a perturbation from source node through the graph.
 * Returns activation values for all nodes (caller must free).
 * `activation` is input/output: initial activation of source, result for all. */
void causal_graph_propagate(const CausalGraph *graph,
                            float *activation,  /* size: max_nodes */
                            size_t source,
                            int steps);         /* propagation depth */

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_CAUSAL_H */
