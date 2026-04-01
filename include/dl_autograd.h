#ifndef DL_AUTOGRAD_H
#define DL_AUTOGRAD_H

#include "dl_tensor.h"

#define DL_MAX_TRACKED (DL_MAX_GRAPH_NODES * 3)

/* Autograd computational graph */
typedef struct {
    DLGraphNode* nodes[DL_MAX_GRAPH_NODES];
    int n_nodes;
    bool no_grad;

    /* Track heap pointers from intermediate tensors for bulk free */
    float* tracked_data[DL_MAX_TRACKED];
    int n_tracked;
} DLGraph;

/* Global graph */
extern DLGraph dl_graph;

/* Graph management */
void dl_graph_init(void);
void dl_graph_clear(void);
DLGraphNode* dl_graph_add_node(DLTensor* output, DLTensor** inputs, int n_inputs,
                                DLBackwardFn backward_fn);

/* Backward pass */
void dl_backward(DLTensor* loss);

/* Track a tensor's data for cleanup by graph_clear (use for non-autograd intermediates) */
void dl_graph_track_tensor(DLTensor* t);

/* No-grad context */
void dl_set_no_grad(bool no_grad);
bool dl_is_no_grad(void);

/* ========== Autograd-enabled operations ========== */
/* These wrap the raw ops and record the computation graph */

DLTensor* dl_ag_matmul(DLTensor* a, DLTensor* b);
DLTensor* dl_ag_add(DLTensor* a, DLTensor* b);
DLTensor* dl_ag_mul(DLTensor* a, DLTensor* b);
DLTensor* dl_ag_scale(DLTensor* a, float scalar);
DLTensor* dl_ag_gelu(DLTensor* x);
DLTensor* dl_ag_relu(DLTensor* x);
DLTensor* dl_ag_silu(DLTensor* x);
DLTensor* dl_ag_softmax(DLTensor* x, int dim);
DLTensor* dl_ag_layer_norm(DLTensor* x, DLTensor* gamma, DLTensor* beta, float eps);
DLTensor* dl_ag_embedding(DLTensor* weight, const int* indices, int n);
DLTensor* dl_ag_cross_entropy(DLTensor* logits, const int* targets, int batch, int vocab);
DLTensor* dl_ag_dropout(DLTensor* x, float p, bool training);
DLTensor* dl_ag_transpose(DLTensor* x, int dim0, int dim1);
DLTensor* dl_ag_reshape(DLTensor* x, const int* shape, int ndim);

#endif /* DL_AUTOGRAD_H */
