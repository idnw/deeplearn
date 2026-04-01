#include "dl_autograd.h"
#include "dl_ops.h"

/* Global computational graph */
DLGraph dl_graph = {.n_nodes = 0, .no_grad = false, .n_tracked = 0};

void dl_graph_init(void) {
    dl_graph.n_nodes = 0;
    dl_graph.no_grad = false;
    dl_graph.n_tracked = 0;
}

/* Track an intermediate tensor's data pointer. When graph_clear is called,
 * all tracked data pointers are freed. The tensor itself has owns_data=false
 * so dl_tensor_free won't double-free. */
static void dl_graph_track(DLTensor* t) {
    if (dl_graph.no_grad) return;
    if (t->requires_grad) return; /* parameters own their data */
    if (!t->owns_data) return;    /* already not owning */
    DL_CHECK(dl_graph.n_tracked < DL_MAX_TRACKED, "too many tracked tensors");
    dl_graph.tracked_data[dl_graph.n_tracked++] = t->data;
    t->owns_data = false; /* graph_clear will free this data */
}

void dl_graph_track_tensor(DLTensor* t) {
    dl_graph_track(t);
}

/* Track grad data allocated for intermediate tensors during backward */
void dl_track_grad_data(float* data) {
    if (dl_graph.no_grad || !data) return;
    if (dl_graph.n_tracked < DL_MAX_TRACKED) {
        dl_graph.tracked_data[dl_graph.n_tracked++] = data;
    }
}

void dl_graph_clear(void) {
    /* Free graph nodes */
    for (int i = 0; i < dl_graph.n_nodes; i++) {
        DLGraphNode* node = dl_graph.nodes[i];
        if (!node) continue;
        if (node->output) node->output->graph_node = NULL;
        if (node->extra) free(node->extra);
        free(node);
        dl_graph.nodes[i] = NULL;
    }
    dl_graph.n_nodes = 0;

    /* Free all tracked intermediate data pointers */
    for (int i = 0; i < dl_graph.n_tracked; i++) {
        if (dl_graph.tracked_data[i]) {
            free(dl_graph.tracked_data[i]);
            dl_graph.tracked_data[i] = NULL;
        }
    }
    dl_graph.n_tracked = 0;
}

DLGraphNode* dl_graph_add_node(DLTensor* output, DLTensor** inputs, int n_inputs,
                                DLBackwardFn backward_fn) {
    if (dl_graph.no_grad) return NULL;

    /* Check if any input requires grad */
    bool needs_grad = false;
    for (int i = 0; i < n_inputs; i++) {
        if (inputs[i] && (inputs[i]->requires_grad || inputs[i]->graph_node)) {
            needs_grad = true;
            break;
        }
    }
    if (!needs_grad) return NULL;

    DL_CHECK(dl_graph.n_nodes < DL_MAX_GRAPH_NODES, "graph full");

    DLGraphNode* node = DL_ALLOC(DLGraphNode, 1);
    memset(node, 0, sizeof(DLGraphNode));
    node->output = output;
    node->n_inputs = n_inputs;
    for (int i = 0; i < n_inputs && i < 4; i++) {
        node->inputs[i] = inputs[i];
    }
    node->backward_fn = backward_fn;
    node->visited = false;

    output->graph_node = node;
    dl_graph.nodes[dl_graph.n_nodes++] = node;
    return node;
}

/* Topological sort helper */
static void dl_topo_sort(DLGraphNode* node, DLGraphNode** sorted, int* count) {
    if (!node || node->visited) return;
    node->visited = true;

    for (int i = 0; i < node->n_inputs; i++) {
        if (node->inputs[i] && node->inputs[i]->graph_node) {
            dl_topo_sort(node->inputs[i]->graph_node, sorted, count);
        }
    }
    sorted[(*count)++] = node;
}

void dl_backward(DLTensor* loss) {
    DL_CHECK(loss->graph_node, "loss has no graph node");

    /* Set initial gradient to 1.0 */
    if (!loss->grad) {
        loss->grad = dl_tensor_ones(loss->shape, loss->ndim);
    } else {
        dl_tensor_fill_(loss->grad, 1.0f);
    }

    /* Topological sort */
    DLGraphNode* sorted[DL_MAX_GRAPH_NODES];
    int count = 0;

    /* Reset visited flags */
    for (int i = 0; i < dl_graph.n_nodes; i++) {
        if (dl_graph.nodes[i]) dl_graph.nodes[i]->visited = false;
    }

    dl_topo_sort(loss->graph_node, sorted, &count);

    /* Backward pass in reverse topological order */
    for (int i = count - 1; i >= 0; i--) {
        DLGraphNode* node = sorted[i];
        if (node->backward_fn && node->output->grad) {
            node->backward_fn(node);
        }
    }
}

void dl_set_no_grad(bool no_grad) {
    dl_graph.no_grad = no_grad;
}

bool dl_is_no_grad(void) {
    return dl_graph.no_grad;
}

/* ========== Backward Functions ========== */

/* Sum over batch dimensions to reduce grad to param shape */
static void dl_accumulate_grad(DLTensor* grad_param, DLTensor* grad_computed) {
    if (dl_tensor_shape_eq(grad_param, grad_computed)) {
        dl_tensor_add_(grad_param, grad_computed);
        return;
    }
    /* grad_computed may have extra batch dims: sum them out */
    if (grad_param->size == 0) return;
    int batch_size = grad_computed->size / grad_param->size;
    if (batch_size * grad_param->size == grad_computed->size) {
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < grad_param->size; i++) {
                grad_param->data[i] += grad_computed->data[b * grad_param->size + i];
            }
        }
    } else {
        /* Fallback: element-wise with wrapping */
        for (int i = 0; i < grad_computed->size; i++) {
            grad_param->data[i % grad_param->size] += grad_computed->data[i];
        }
    }
}

static void backward_matmul(DLGraphNode* node) {
    DLTensor* a = node->inputs[0];
    DLTensor* b = node->inputs[1];
    DLTensor* grad_out = node->output->grad;

    if (a->requires_grad || a->graph_node) {
        DLTensor* bt = dl_tensor_transpose(b, b->ndim - 2, b->ndim - 1);
        DLTensor* bt_c = dl_tensor_contiguous(bt);
        DLTensor* grad_a = dl_matmul(grad_out, bt_c);
        DLTensor* ga = dl_tensor_ensure_grad(a);
        dl_accumulate_grad(ga, grad_a);
        dl_tensor_free(bt);
        dl_tensor_free(bt_c);
        dl_tensor_free(grad_a);
    }

    if (b->requires_grad || b->graph_node) {
        DLTensor* at = dl_tensor_transpose(a, a->ndim - 2, a->ndim - 1);
        DLTensor* at_c = dl_tensor_contiguous(at);
        DLTensor* grad_b = dl_matmul(at_c, grad_out);
        DLTensor* gb = dl_tensor_ensure_grad(b);
        dl_accumulate_grad(gb, grad_b);
        dl_tensor_free(at);
        dl_tensor_free(at_c);
        dl_tensor_free(grad_b);
    }
}

static void backward_add(DLGraphNode* node) {
    DLTensor* a = node->inputs[0];
    DLTensor* b = node->inputs[1];
    DLTensor* grad_out = node->output->grad;

    if (a->requires_grad || a->graph_node) {
        DLTensor* ga = dl_tensor_ensure_grad(a);
        if (dl_tensor_shape_eq(a, node->output)) {
            dl_tensor_add_(ga, grad_out);
        } else {
            /* Need to reduce gradient for broadcast */
            for (int d = 0; d < node->output->ndim; d++) {
                int ad = d < node->output->ndim - a->ndim ? 1 :
                         a->shape[d - (node->output->ndim - a->ndim)];
                if (ad == 1 && node->output->shape[d] > 1) {
                    DLTensor* sum = dl_tensor_sum(grad_out, d, true);
                    /* Simplified - accumulate into grad */
                    for (int i = 0; i < ga->size && i < sum->size; i++)
                        ga->data[i] += sum->data[i];
                    dl_tensor_free(sum);
                    break;
                }
            }
        }
    }
    if (b->requires_grad || b->graph_node) {
        DLTensor* gb = dl_tensor_ensure_grad(b);
        if (dl_tensor_shape_eq(b, node->output)) {
            dl_tensor_add_(gb, grad_out);
        } else {
            for (int d = 0; d < node->output->ndim; d++) {
                int bd = d < node->output->ndim - b->ndim ? 1 :
                         b->shape[d - (node->output->ndim - b->ndim)];
                if (bd == 1 && node->output->shape[d] > 1) {
                    DLTensor* sum = dl_tensor_sum(grad_out, d, true);
                    for (int i = 0; i < gb->size && i < sum->size; i++)
                        gb->data[i] += sum->data[i];
                    dl_tensor_free(sum);
                    break;
                }
            }
        }
    }
}

static void backward_mul(DLGraphNode* node) {
    DLTensor* a = node->inputs[0];
    DLTensor* b = node->inputs[1];
    DLTensor* grad_out = node->output->grad;

    if (a->requires_grad || a->graph_node) {
        DLTensor* grad_a = dl_tensor_mul(grad_out, b);
        DLTensor* ga = dl_tensor_ensure_grad(a);
        if (dl_tensor_shape_eq(ga, grad_a)) {
            dl_tensor_add_(ga, grad_a);
        } else {
            for (int i = 0; i < ga->size; i++) ga->data[i] += grad_a->data[i % grad_a->size];
        }
        dl_tensor_free(grad_a);
    }
    if (b->requires_grad || b->graph_node) {
        DLTensor* grad_b = dl_tensor_mul(grad_out, a);
        DLTensor* gb = dl_tensor_ensure_grad(b);
        if (dl_tensor_shape_eq(gb, grad_b)) {
            dl_tensor_add_(gb, grad_b);
        } else {
            for (int i = 0; i < gb->size; i++) gb->data[i] += grad_b->data[i % grad_b->size];
        }
        dl_tensor_free(grad_b);
    }
}

static void backward_scale(DLGraphNode* node) {
    DLTensor* a = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    float scalar = node->extra_float[0];

    if (a->requires_grad || a->graph_node) {
        DLTensor* grad_a = dl_tensor_scale(grad_out, scalar);
        DLTensor* ga = dl_tensor_ensure_grad(a);
        dl_accumulate_grad(ga, grad_a);
        dl_tensor_free(grad_a);
    }
}

static void backward_gelu(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;

    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        DLTensor* cx = dl_tensor_contiguous(x);
        for (int i = 0; i < x->size; i++) {
            float xi = cx->data[i];
            float x3 = xi * xi * xi;
            float inner = 0.7978845608f * (xi + 0.044715f * x3);
            float tanh_val = tanhf(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
            float grad = 0.5f * (1.0f + tanh_val) + 0.5f * xi * sech2 * d_inner;
            gx->data[i] += grad_out->data[i] * grad;
        }
        dl_tensor_free(cx);
    }
}

static void backward_relu(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        DLTensor* cx = dl_tensor_contiguous(x);
        for (int i = 0; i < x->size; i++) {
            gx->data[i] += (cx->data[i] > 0) ? grad_out->data[i] : 0;
        }
        dl_tensor_free(cx);
    }
}

static void backward_silu(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        DLTensor* cx = dl_tensor_contiguous(x);
        for (int i = 0; i < x->size; i++) {
            float xi = cx->data[i];
            float sig = 1.0f / (1.0f + expf(-xi));
            float grad = sig + xi * sig * (1.0f - sig);
            gx->data[i] += grad_out->data[i] * grad;
        }
        dl_tensor_free(cx);
    }
}

static void backward_softmax(DLGraphNode* node) {
    DLTensor* grad_out = node->output->grad;
    DLTensor* out = node->output;
    DLTensor* x = node->inputs[0];

    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        int dim = node->extra_int[0];
        int outer = 1, inner = 1, axis_size = out->shape[dim];
        for (int d = 0; d < dim; d++) outer *= out->shape[d];
        for (int d = dim + 1; d < out->ndim; d++) inner *= out->shape[d];

        for (int o = 0; o < outer; o++) {
            for (int ii = 0; ii < inner; ii++) {
                /* sum = sum(grad_out * softmax_out) */
                float sum = 0.0f;
                for (int a = 0; a < axis_size; a++) {
                    int idx = (o * axis_size + a) * inner + ii;
                    sum += grad_out->data[idx] * out->data[idx];
                }
                for (int a = 0; a < axis_size; a++) {
                    int idx = (o * axis_size + a) * inner + ii;
                    gx->data[idx] += out->data[idx] * (grad_out->data[idx] - sum);
                }
            }
        }
    }
}

static void backward_layer_norm(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* gamma = node->inputs[1];
    DLTensor* grad_out = node->output->grad;
    float eps = node->extra_float[0];

    int norm_size = x->shape[x->ndim - 1];
    int batch_size = x->size / norm_size;

    DLTensor* cx = dl_tensor_contiguous(x);

    if (gamma && (gamma->requires_grad || gamma->graph_node)) {
        /* dL/dgamma = sum over batch of (grad_out * normalized_x) */
        DLTensor* gg = dl_tensor_ensure_grad(gamma);
        for (int b = 0; b < batch_size; b++) {
            float* row = cx->data + b * norm_size;
            float mean = 0.0f;
            for (int i = 0; i < norm_size; i++) mean += row[i];
            mean /= norm_size;
            float var = 0.0f;
            for (int i = 0; i < norm_size; i++) {
                float d = row[i] - mean;
                var += d * d;
            }
            var /= norm_size;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int i = 0; i < norm_size; i++) {
                float norm_x = (row[i] - mean) * inv_std;
                gg->data[i] += grad_out->data[b * norm_size + i] * norm_x;
            }
        }
    }

    DLTensor* beta = node->inputs[2];
    if (beta && (beta->requires_grad || beta->graph_node)) {
        DLTensor* gb = dl_tensor_ensure_grad(beta);
        for (int b = 0; b < batch_size; b++) {
            for (int i = 0; i < norm_size; i++) {
                gb->data[i] += grad_out->data[b * norm_size + i];
            }
        }
    }

    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        for (int b = 0; b < batch_size; b++) {
            float* row = cx->data + b * norm_size;
            float* grow = grad_out->data + b * norm_size;
            float mean = 0.0f;
            for (int i = 0; i < norm_size; i++) mean += row[i];
            mean /= norm_size;
            float var = 0.0f;
            for (int i = 0; i < norm_size; i++) {
                float d = row[i] - mean;
                var += d * d;
            }
            var /= norm_size;
            float inv_std = 1.0f / sqrtf(var + eps);

            /* Compute gradient through layer norm */
            float sum_dy = 0.0f, sum_dy_x = 0.0f;
            for (int i = 0; i < norm_size; i++) {
                float dy = grow[i];
                if (gamma) dy *= gamma->data[i];
                sum_dy += dy;
                sum_dy_x += dy * (row[i] - mean);
            }
            for (int i = 0; i < norm_size; i++) {
                float dy = grow[i];
                if (gamma) dy *= gamma->data[i];
                float dx = inv_std * (dy - (sum_dy + (row[i] - mean) * inv_std * inv_std * sum_dy_x) / norm_size);
                gx->data[b * norm_size + i] += dx;
            }
        }
    }
    dl_tensor_free(cx);
}

static void backward_embedding(DLGraphNode* node) {
    DLTensor* weight = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    int n = node->extra_int[0];
    int* indices = (int*)node->extra;

    if (weight->requires_grad) {
        DLTensor* gw = dl_tensor_ensure_grad(weight);
        int embed_dim = weight->shape[1];
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            for (int j = 0; j < embed_dim; j++) {
                gw->data[idx * embed_dim + j] += grad_out->data[i * embed_dim + j];
            }
        }
    }
}

static void backward_cross_entropy(DLGraphNode* node) {
    DLTensor* logits = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    int batch_size = node->extra_int[0];
    int vocab_size = node->extra_int[1];
    int* targets = (int*)node->extra;

    if (logits->requires_grad || logits->graph_node) {
        DLTensor* gl = dl_tensor_ensure_grad(logits);
        /* Softmax - one_hot(target) */
        DLTensor* probs = dl_softmax(logits, -1);
        float scale = grad_out->data[0] / batch_size;
        for (int b = 0; b < batch_size; b++) {
            for (int v = 0; v < vocab_size; v++) {
                float grad = probs->data[b * vocab_size + v];
                if (v == targets[b]) grad -= 1.0f;
                gl->data[b * vocab_size + v] += grad * scale;
            }
        }
        dl_tensor_free(probs);
    }
}

static void backward_dropout(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        float* mask = (float*)node->extra;
        for (int i = 0; i < x->size; i++) {
            gx->data[i] += grad_out->data[i] * mask[i];
        }
    }
}

static void backward_transpose(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    if (x->requires_grad || x->graph_node) {
        int dim0 = node->extra_int[0];
        int dim1 = node->extra_int[1];
        DLTensor* gt = dl_tensor_transpose(grad_out, dim0, dim1);
        DLTensor* gtc = dl_tensor_contiguous(gt);
        DLTensor* gx = dl_tensor_ensure_grad(x);
        dl_accumulate_grad(gx, gtc);
        dl_tensor_free(gt);
        dl_tensor_free(gtc);
    }
}

static void backward_reshape(DLGraphNode* node) {
    DLTensor* x = node->inputs[0];
    DLTensor* grad_out = node->output->grad;
    if (x->requires_grad || x->graph_node) {
        DLTensor* gx = dl_tensor_ensure_grad(x);
        /* grad_out has same number of elements, just different shape */
        DLTensor* gc = dl_tensor_contiguous(grad_out);
        for (int i = 0; i < gx->size && i < gc->size; i++) {
            gx->data[i] += gc->data[i];
        }
        dl_tensor_free(gc);
    }
}

/* ========== Autograd-Enabled Operations ========== */

DLTensor* dl_ag_matmul(DLTensor* a, DLTensor* b) {
    DLTensor* out = dl_matmul(a, b);
    DLTensor* inputs[] = {a, b};
    dl_graph_add_node(out, inputs, 2, backward_matmul);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_add(DLTensor* a, DLTensor* b) {
    DLTensor* out = dl_tensor_add(a, b);
    DLTensor* inputs[] = {a, b};
    dl_graph_add_node(out, inputs, 2, backward_add);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_mul(DLTensor* a, DLTensor* b) {
    DLTensor* out = dl_tensor_mul(a, b);
    DLTensor* inputs[] = {a, b};
    dl_graph_add_node(out, inputs, 2, backward_mul);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_scale(DLTensor* a, float scalar) {
    DLTensor* out = dl_tensor_scale(a, scalar);
    DLTensor* inputs[] = {a};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_scale);
    if (node) node->extra_float[0] = scalar;
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_gelu(DLTensor* x) {
    DLTensor* out = dl_gelu(x);
    DLTensor* inputs[] = {x};
    dl_graph_add_node(out, inputs, 1, backward_gelu);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_relu(DLTensor* x) {
    DLTensor* out = dl_relu(x);
    DLTensor* inputs[] = {x};
    dl_graph_add_node(out, inputs, 1, backward_relu);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_silu(DLTensor* x) {
    DLTensor* out = dl_silu(x);
    DLTensor* inputs[] = {x};
    dl_graph_add_node(out, inputs, 1, backward_silu);
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_softmax(DLTensor* x, int dim) {
    if (dim < 0) dim += x->ndim;
    DLTensor* out = dl_softmax(x, dim);
    DLTensor* inputs[] = {x};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_softmax);
    if (node) node->extra_int[0] = dim;
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_layer_norm(DLTensor* x, DLTensor* gamma, DLTensor* beta, float eps) {
    DLTensor* out = dl_layer_norm(x, gamma, beta, eps);
    DLTensor* inputs[] = {x, gamma, beta};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 3, backward_layer_norm);
    if (node) node->extra_float[0] = eps;
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_embedding(DLTensor* weight, const int* indices, int n) {
    DLTensor* out = dl_embedding_forward(weight, indices, n);
    DLTensor* inputs[] = {weight};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_embedding);
    if (node) {
        node->extra_int[0] = n;
        node->extra = dl_malloc(n * sizeof(int));
        memcpy(node->extra, indices, n * sizeof(int));
    }
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_cross_entropy(DLTensor* logits, const int* targets, int batch, int vocab) {
    DLTensor* out = dl_cross_entropy_loss(logits, targets, batch, vocab);
    DLTensor* inputs[] = {logits};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_cross_entropy);
    if (node) {
        node->extra_int[0] = batch;
        node->extra_int[1] = vocab;
        node->extra = dl_malloc(batch * sizeof(int));
        memcpy(node->extra, targets, batch * sizeof(int));
    }
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_dropout(DLTensor* x, float p, bool training) {
    if (!training || p == 0.0f) {
        DLTensor* out = dl_tensor_clone(x);
        dl_graph_track(out);
        return out;
    }

    dl_rng_ensure_init();
    DLTensor* out = dl_tensor_create(x->shape, x->ndim);
    float scale = 1.0f / (1.0f - p);
    float* mask = DL_ALLOC(float, x->size);

    DLTensor* cx = dl_tensor_contiguous(x);
    for (int i = 0; i < x->size; i++) {
        mask[i] = (dl_rng_float(&dl_global_rng) > p) ? scale : 0.0f;
        out->data[i] = cx->data[i] * mask[i];
    }
    dl_tensor_free(cx);

    DLTensor* inputs[] = {x};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_dropout);
    if (node) {
        node->extra = mask;
    } else {
        free(mask);
    }
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_transpose(DLTensor* x, int dim0, int dim1) {
    DLTensor* t = dl_tensor_transpose(x, dim0, dim1);
    DLTensor* out = dl_tensor_contiguous(t);
    dl_tensor_free(t);

    DLTensor* inputs[] = {x};
    DLGraphNode* node = dl_graph_add_node(out, inputs, 1, backward_transpose);
    if (node) {
        node->extra_int[0] = dim0;
        node->extra_int[1] = dim1;
    }
    dl_graph_track(out);
    return out;
}

DLTensor* dl_ag_reshape(DLTensor* x, const int* shape, int ndim) {
    DLTensor* cx = dl_tensor_contiguous(x);
    DLTensor* out = dl_tensor_create(shape, ndim);
    memcpy(out->data, cx->data, cx->size * sizeof(float));
    dl_tensor_free(cx);

    DLTensor* inputs[] = {x};
    dl_graph_add_node(out, inputs, 1, backward_reshape);
    dl_graph_track(out);
    return out;
}
