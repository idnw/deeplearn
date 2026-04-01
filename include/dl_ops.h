#ifndef DL_OPS_H
#define DL_OPS_H

#include "dl_tensor.h"

/* Matrix multiplication: (M,K) x (K,N) -> (M,N) */
DLTensor* dl_matmul(DLTensor* a, DLTensor* b);

/* Batched matmul: (...,M,K) x (...,K,N) -> (...,M,N) */
DLTensor* dl_bmm(DLTensor* a, DLTensor* b);

/* Softmax along last dimension */
DLTensor* dl_softmax(DLTensor* t, int dim);

/* Log-softmax along dimension */
DLTensor* dl_log_softmax(DLTensor* t, int dim);

/* Layer normalization */
DLTensor* dl_layer_norm(DLTensor* x, DLTensor* gamma, DLTensor* beta, float eps);

/* Activations */
DLTensor* dl_gelu(DLTensor* t);
DLTensor* dl_relu(DLTensor* t);
DLTensor* dl_silu(DLTensor* t);

/* Loss functions */
DLTensor* dl_cross_entropy_loss(DLTensor* logits, const int* targets, int batch_size, int vocab_size);

/* Embedding lookup */
DLTensor* dl_embedding_forward(DLTensor* weight, const int* indices, int n);

/* Utility ops */
DLTensor* dl_concat(DLTensor** tensors, int n, int dim);
DLTensor* dl_split(DLTensor* t, int n_splits, int dim, int split_idx);

#endif /* DL_OPS_H */
