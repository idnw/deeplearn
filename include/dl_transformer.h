#ifndef DL_TRANSFORMER_H
#define DL_TRANSFORMER_H

#include "dl_nn.h"

/* Transformer configuration */
typedef struct {
    int vocab_size;
    int max_seq_len;
    int n_layers;
    int n_heads;
    int d_model;
    int d_ff;       /* feed-forward intermediate size, typically 4 * d_model */
    float dropout_p;
    float layer_norm_eps;
} DLTransformerConfig;

/* Multi-Head Attention */
typedef struct {
    DLLinear* q_proj;
    DLLinear* k_proj;
    DLLinear* v_proj;
    DLLinear* o_proj;
    DLDropout* attn_dropout;
    DLDropout* resid_dropout;
    int n_heads;
    int d_model;
    int head_dim;
} DLMultiHeadAttention;

DLMultiHeadAttention* dl_mha_create(int d_model, int n_heads, float dropout_p);
void dl_mha_free(DLMultiHeadAttention* mha);
DLTensor* dl_mha_forward(DLMultiHeadAttention* mha, DLTensor* x,
                          bool causal_mask, bool training);
void dl_mha_add_params(DLMultiHeadAttention* mha, DLParamList* params);

/* Transformer Block (Pre-norm) */
typedef struct {
    DLLayerNorm* ln1;
    DLMultiHeadAttention* attn;
    DLLayerNorm* ln2;
    DLLinear* ff1;     /* d_model -> d_ff */
    DLLinear* ff2;     /* d_ff -> d_model */
    DLDropout* ff_dropout;
    int d_model;
    int d_ff;
} DLTransformerBlock;

DLTransformerBlock* dl_block_create(int d_model, int n_heads, int d_ff, float dropout_p, float ln_eps);
void dl_block_free(DLTransformerBlock* block);
DLTensor* dl_block_forward(DLTransformerBlock* block, DLTensor* x, bool training);
void dl_block_add_params(DLTransformerBlock* block, DLParamList* params);

/* Full Transformer Model (GPT-style decoder) */
typedef struct {
    DLTransformerConfig config;
    DLEmbedding* token_embed;
    DLEmbedding* pos_embed;
    DLTransformerBlock** blocks;
    DLLayerNorm* final_ln;
    DLLinear* lm_head;    /* language model output head */
    DLDropout* embed_dropout;
    DLParamList* params;
    bool training;
} DLTransformerModel;

DLTransformerModel* dl_transformer_create(DLTransformerConfig config);
void dl_transformer_free(DLTransformerModel* model);
DLTensor* dl_transformer_forward(DLTransformerModel* model, const int* tokens,
                                  int batch_size, int seq_len);
DLTensor* dl_transformer_loss(DLTransformerModel* model, const int* tokens,
                               int batch_size, int seq_len);
void dl_transformer_set_training(DLTransformerModel* model, bool training);

/* Inference helpers */
int dl_transformer_generate_next(DLTransformerModel* model, const int* tokens,
                                  int seq_len, float temperature, int top_k);

#endif /* DL_TRANSFORMER_H */
