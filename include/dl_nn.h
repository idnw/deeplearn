#ifndef DL_NN_H
#define DL_NN_H

#include "dl_autograd.h"

/* Linear layer */
typedef struct {
    DLTensor* weight; /* (out_features, in_features) */
    DLTensor* bias;   /* (out_features) or NULL */
    int in_features;
    int out_features;
} DLLinear;

DLLinear* dl_linear_create(int in_features, int out_features, bool use_bias);
void dl_linear_free(DLLinear* layer);
DLTensor* dl_linear_forward(DLLinear* layer, DLTensor* input);

/* Embedding layer */
typedef struct {
    DLTensor* weight; /* (vocab_size, embed_dim) */
    int vocab_size;
    int embed_dim;
} DLEmbedding;

DLEmbedding* dl_embedding_create(int vocab_size, int embed_dim);
void dl_embedding_free(DLEmbedding* layer);
DLTensor* dl_embedding_lookup(DLEmbedding* layer, const int* indices, int n);

/* LayerNorm */
typedef struct {
    DLTensor* gamma; /* (normalized_shape) */
    DLTensor* beta;  /* (normalized_shape) */
    int normalized_shape;
    float eps;
} DLLayerNorm;

DLLayerNorm* dl_layernorm_create(int normalized_shape, float eps);
void dl_layernorm_free(DLLayerNorm* layer);
DLTensor* dl_layernorm_forward(DLLayerNorm* layer, DLTensor* input);

/* Dropout */
typedef struct {
    float p;
    bool training;
} DLDropout;

DLDropout* dl_dropout_create(float p);
void dl_dropout_free(DLDropout* layer);
DLTensor* dl_dropout_forward(DLDropout* layer, DLTensor* input);

/* Parameter collection */
typedef struct {
    DLTensor** params;
    int n_params;
    int capacity;
} DLParamList;

DLParamList* dl_paramlist_create(int initial_capacity);
void dl_paramlist_free(DLParamList* list);
void dl_paramlist_add(DLParamList* list, DLTensor* param);
void dl_paramlist_add_linear(DLParamList* list, DLLinear* layer);
void dl_paramlist_add_embedding(DLParamList* list, DLEmbedding* layer);
void dl_paramlist_add_layernorm(DLParamList* list, DLLayerNorm* layer);
void dl_paramlist_zero_grad(DLParamList* list);
int dl_paramlist_total_params(DLParamList* list);

#endif /* DL_NN_H */
