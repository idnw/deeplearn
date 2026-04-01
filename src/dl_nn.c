#include "dl_nn.h"

/* ========== Linear Layer ========== */

DLLinear* dl_linear_create(int in_features, int out_features, bool use_bias) {
    DLLinear* layer = DL_ALLOC(DLLinear, 1);
    layer->in_features = in_features;
    layer->out_features = out_features;

    /* Kaiming initialization */
    int shape[2] = {out_features, in_features};
    float std = sqrtf(2.0f / in_features);
    layer->weight = dl_tensor_randn(shape, 2, 0.0f, std);
    dl_tensor_set_requires_grad(layer->weight, true);

    if (use_bias) {
        int bshape[1] = {out_features};
        layer->bias = dl_tensor_zeros(bshape, 1);
        dl_tensor_set_requires_grad(layer->bias, true);
    } else {
        layer->bias = NULL;
    }
    return layer;
}

void dl_linear_free(DLLinear* layer) {
    if (!layer) return;
    dl_tensor_free(layer->weight);
    if (layer->bias) dl_tensor_free(layer->bias);
    free(layer);
}

DLTensor* dl_linear_forward(DLLinear* layer, DLTensor* input) {
    /* input: (..., in_features) -> output: (..., out_features) */
    /* y = x @ W^T + b */
    DLTensor* wt = dl_tensor_transpose(layer->weight, 0, 1);
    DLTensor* wt_c = dl_tensor_contiguous(wt);
    dl_tensor_free(wt); /* view only, safe to free */
    if (!dl_is_no_grad()) {
        /* Track wt_c so it stays alive for backward_matmul */
        dl_graph_track_tensor(wt_c);
    }

    /* Handle batched input */
    int batch_size = 1;
    for (int d = 0; d < input->ndim - 1; d++) batch_size *= input->shape[d];

    int in_shape[2] = {batch_size, layer->in_features};
    DLTensor* flat_input = dl_ag_reshape(input, in_shape, 2);
    DLTensor* mm_out = dl_ag_matmul(flat_input, wt_c);

    DLTensor* out = mm_out;
    if (layer->bias) {
        out = dl_ag_add(mm_out, layer->bias);
    }

    /* Reshape back */
    if (input->ndim > 2) {
        int out_shape[DL_MAX_DIMS];
        for (int d = 0; d < input->ndim - 1; d++) out_shape[d] = input->shape[d];
        out_shape[input->ndim - 1] = layer->out_features;
        DLTensor* reshaped = dl_ag_reshape(out, out_shape, input->ndim);
        if (dl_is_no_grad()) {
            dl_tensor_free(wt_c);
            dl_tensor_free(out);
            dl_tensor_free(flat_input);
            if (out != mm_out) dl_tensor_free(mm_out);
        }
        return reshaped;
    }

    if (dl_is_no_grad()) {
        dl_tensor_free(wt_c);
        dl_tensor_free(flat_input);
        if (out != mm_out) dl_tensor_free(mm_out);
    }
    return out;
}

/* ========== Embedding Layer ========== */

DLEmbedding* dl_embedding_create(int vocab_size, int embed_dim) {
    DLEmbedding* layer = DL_ALLOC(DLEmbedding, 1);
    layer->vocab_size = vocab_size;
    layer->embed_dim = embed_dim;

    int shape[2] = {vocab_size, embed_dim};
    layer->weight = dl_tensor_randn(shape, 2, 0.0f, 0.02f);
    dl_tensor_set_requires_grad(layer->weight, true);
    return layer;
}

void dl_embedding_free(DLEmbedding* layer) {
    if (!layer) return;
    dl_tensor_free(layer->weight);
    free(layer);
}

DLTensor* dl_embedding_lookup(DLEmbedding* layer, const int* indices, int n) {
    return dl_ag_embedding(layer->weight, indices, n);
}

/* ========== LayerNorm ========== */

DLLayerNorm* dl_layernorm_create(int normalized_shape, float eps) {
    DLLayerNorm* layer = DL_ALLOC(DLLayerNorm, 1);
    layer->normalized_shape = normalized_shape;
    layer->eps = eps;

    int shape[1] = {normalized_shape};
    layer->gamma = dl_tensor_ones(shape, 1);
    dl_tensor_set_requires_grad(layer->gamma, true);
    layer->beta = dl_tensor_zeros(shape, 1);
    dl_tensor_set_requires_grad(layer->beta, true);
    return layer;
}

void dl_layernorm_free(DLLayerNorm* layer) {
    if (!layer) return;
    dl_tensor_free(layer->gamma);
    dl_tensor_free(layer->beta);
    free(layer);
}

DLTensor* dl_layernorm_forward(DLLayerNorm* layer, DLTensor* input) {
    return dl_ag_layer_norm(input, layer->gamma, layer->beta, layer->eps);
}

/* ========== Dropout ========== */

DLDropout* dl_dropout_create(float p) {
    DLDropout* layer = DL_ALLOC(DLDropout, 1);
    layer->p = p;
    layer->training = true;
    return layer;
}

void dl_dropout_free(DLDropout* layer) {
    if (layer) free(layer);
}

DLTensor* dl_dropout_forward(DLDropout* layer, DLTensor* input) {
    return dl_ag_dropout(input, layer->p, layer->training);
}

/* ========== Parameter Collection ========== */

DLParamList* dl_paramlist_create(int initial_capacity) {
    DLParamList* list = DL_ALLOC(DLParamList, 1);
    list->params = DL_ALLOC(DLTensor*, initial_capacity);
    list->n_params = 0;
    list->capacity = initial_capacity;
    return list;
}

void dl_paramlist_free(DLParamList* list) {
    if (!list) return;
    free(list->params);
    free(list);
}

void dl_paramlist_add(DLParamList* list, DLTensor* param) {
    if (list->n_params >= list->capacity) {
        list->capacity *= 2;
        list->params = (DLTensor**)dl_realloc(list->params,
                                               list->capacity * sizeof(DLTensor*));
    }
    list->params[list->n_params++] = param;
}

void dl_paramlist_add_linear(DLParamList* list, DLLinear* layer) {
    dl_paramlist_add(list, layer->weight);
    if (layer->bias) dl_paramlist_add(list, layer->bias);
}

void dl_paramlist_add_embedding(DLParamList* list, DLEmbedding* layer) {
    dl_paramlist_add(list, layer->weight);
}

void dl_paramlist_add_layernorm(DLParamList* list, DLLayerNorm* layer) {
    dl_paramlist_add(list, layer->gamma);
    dl_paramlist_add(list, layer->beta);
}

void dl_paramlist_zero_grad(DLParamList* list) {
    for (int i = 0; i < list->n_params; i++) {
        dl_tensor_zero_grad(list->params[i]);
    }
}

int dl_paramlist_total_params(DLParamList* list) {
    int total = 0;
    for (int i = 0; i < list->n_params; i++) {
        total += list->params[i]->size;
    }
    return total;
}
