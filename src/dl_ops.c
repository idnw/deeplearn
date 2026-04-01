#include "dl_ops.h"

/* ========== Matrix Multiplication ========== */

/* Tiled matmul for cache efficiency */
#define TILE_SIZE 32

static void dl_matmul_tiled(float* C, const float* A, const float* B,
                             int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
                int imax = i0 + TILE_SIZE < M ? i0 + TILE_SIZE : M;
                int jmax = j0 + TILE_SIZE < N ? j0 + TILE_SIZE : N;
                int kmax = k0 + TILE_SIZE < K ? k0 + TILE_SIZE : K;
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        float a_ik = A[i * K + k];
                        int j = j0;
#if DL_USE_AVX2
                        __m256 va = _mm256_set1_ps(a_ik);
                        for (; j + 7 < jmax; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                            vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));
                            _mm256_storeu_ps(&C[i * N + j], vc);
                        }
#elif DL_USE_SSE2
                        __m128 va = _mm_set1_ps(a_ik);
                        for (; j + 3 < jmax; j += 4) {
                            __m128 vb = _mm_loadu_ps(&B[k * N + j]);
                            __m128 vc = _mm_loadu_ps(&C[i * N + j]);
                            vc = _mm_add_ps(vc, _mm_mul_ps(va, vb));
                            _mm_storeu_ps(&C[i * N + j], vc);
                        }
#endif
                        for (; j < jmax; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

DLTensor* dl_matmul(DLTensor* a, DLTensor* b) {
    DL_CHECK(a->ndim >= 1 && b->ndim >= 1, "matmul requires at least 1D tensors");

    /* Handle 2D x 2D */
    if (a->ndim == 2 && b->ndim == 2) {
        DL_CHECK(a->shape[1] == b->shape[0], "matmul shape mismatch");
        int M = a->shape[0], K = a->shape[1], N = b->shape[1];
        int out_shape[2] = {M, N};
        DLTensor* out = dl_tensor_create(out_shape, 2);

        DLTensor* ca = dl_tensor_contiguous(a);
        DLTensor* cb = dl_tensor_contiguous(b);
        dl_matmul_tiled(out->data, ca->data, cb->data, M, K, N);
        dl_tensor_free(ca);
        dl_tensor_free(cb);
        return out;
    }

    /* Handle batched: (..., M, K) x (..., K, N) */
    DL_CHECK(a->ndim >= 2 && b->ndim >= 2, "batched matmul needs >= 2D");
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];
    DL_CHECK(b->shape[b->ndim - 2] == K, "matmul inner dim mismatch");

    /* Compute batch dimensions */
    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int out_shape[DL_MAX_DIMS];
    int batch_size = 1;
    for (int i = 0; i < max_ndim - 2; i++) {
        int da = i < a->ndim - 2 ? a->shape[i] : 1;
        int db = i < b->ndim - 2 ? b->shape[i] : 1;
        DL_CHECK(da == db || da == 1 || db == 1, "batch dim mismatch");
        out_shape[i] = da > db ? da : db;
        batch_size *= out_shape[i];
    }
    out_shape[max_ndim - 2] = M;
    out_shape[max_ndim - 1] = N;

    DLTensor* out = dl_tensor_create(out_shape, max_ndim);
    DLTensor* ca = dl_tensor_contiguous(a);
    DLTensor* cb = dl_tensor_contiguous(b);

    int a_batch_stride = M * K;
    int b_batch_stride = K * N;
    int o_batch_stride = M * N;

    for (int batch = 0; batch < batch_size; batch++) {
        dl_matmul_tiled(out->data + batch * o_batch_stride,
                        ca->data + batch * a_batch_stride,
                        cb->data + batch * b_batch_stride,
                        M, K, N);
    }

    dl_tensor_free(ca);
    dl_tensor_free(cb);
    return out;
}

DLTensor* dl_bmm(DLTensor* a, DLTensor* b) {
    return dl_matmul(a, b);
}

/* ========== Softmax ========== */

DLTensor* dl_softmax(DLTensor* t, int dim) {
    if (dim < 0) dim += t->ndim;
    DL_CHECK(dim >= 0 && dim < t->ndim, "invalid softmax dim");

    DLTensor* out = dl_tensor_clone(t);
    int outer = 1, inner = 1, axis_size = t->shape[dim];
    for (int d = 0; d < dim; d++) outer *= t->shape[d];
    for (int d = dim + 1; d < t->ndim; d++) inner *= t->shape[d];

    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            /* Find max for numerical stability */
            float max_val = -FLT_MAX;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                if (out->data[idx] > max_val) max_val = out->data[idx];
            }
            /* Exp and sum */
            float sum = 0.0f;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                out->data[idx] = expf(out->data[idx] - max_val);
                sum += out->data[idx];
            }
            /* Normalize */
            float inv_sum = 1.0f / sum;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                out->data[idx] *= inv_sum;
            }
        }
    }
    return out;
}

DLTensor* dl_log_softmax(DLTensor* t, int dim) {
    if (dim < 0) dim += t->ndim;
    DLTensor* out = dl_tensor_clone(t);
    int outer = 1, inner = 1, axis_size = t->shape[dim];
    for (int d = 0; d < dim; d++) outer *= t->shape[d];
    for (int d = dim + 1; d < t->ndim; d++) inner *= t->shape[d];

    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            float max_val = -FLT_MAX;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                if (out->data[idx] > max_val) max_val = out->data[idx];
            }
            float sum = 0.0f;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                sum += expf(out->data[idx] - max_val);
            }
            float log_sum = logf(sum) + max_val;
            for (int a = 0; a < axis_size; a++) {
                int idx = (o * axis_size + a) * inner + i;
                out->data[idx] -= log_sum;
            }
        }
    }
    return out;
}

/* ========== Layer Normalization ========== */

DLTensor* dl_layer_norm(DLTensor* x, DLTensor* gamma, DLTensor* beta, float eps) {
    DL_CHECK(x->ndim >= 1, "layernorm needs at least 1D");
    int norm_size = x->shape[x->ndim - 1];
    int batch_size = x->size / norm_size;

    DLTensor* out = dl_tensor_create(x->shape, x->ndim);
    DLTensor* cx = dl_tensor_contiguous(x);

    for (int b = 0; b < batch_size; b++) {
        float* row = cx->data + b * norm_size;
        float* orow = out->data + b * norm_size;

        /* Compute mean */
        float mean = 0.0f;
        for (int i = 0; i < norm_size; i++) mean += row[i];
        mean /= norm_size;

        /* Compute variance */
        float var = 0.0f;
        for (int i = 0; i < norm_size; i++) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= norm_size;

        /* Normalize */
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < norm_size; i++) {
            float normalized = (row[i] - mean) * inv_std;
            orow[i] = normalized;
            if (gamma) orow[i] *= gamma->data[i];
            if (beta) orow[i] += beta->data[i];
        }
    }
    dl_tensor_free(cx);
    return out;
}

/* ========== Activations ========== */

DLTensor* dl_gelu(DLTensor* t) {
    DLTensor* out = dl_tensor_create(t->shape, t->ndim);
    DLTensor* ct = dl_tensor_contiguous(t);
    for (int i = 0; i < out->size; i++) {
        float x = ct->data[i];
        /* Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        out->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
    dl_tensor_free(ct);
    return out;
}

DLTensor* dl_relu(DLTensor* t) {
    DLTensor* out = dl_tensor_create(t->shape, t->ndim);
    DLTensor* ct = dl_tensor_contiguous(t);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = ct->data[i] > 0 ? ct->data[i] : 0;
    }
    dl_tensor_free(ct);
    return out;
}

DLTensor* dl_silu(DLTensor* t) {
    DLTensor* out = dl_tensor_create(t->shape, t->ndim);
    DLTensor* ct = dl_tensor_contiguous(t);
    for (int i = 0; i < out->size; i++) {
        float x = ct->data[i];
        out->data[i] = x / (1.0f + expf(-x));
    }
    dl_tensor_free(ct);
    return out;
}

/* ========== Loss Functions ========== */

DLTensor* dl_cross_entropy_loss(DLTensor* logits, const int* targets,
                                 int batch_size, int vocab_size) {
    DL_CHECK(logits->ndim == 2, "cross entropy expects 2D logits");
    DL_CHECK(logits->shape[0] == batch_size && logits->shape[1] == vocab_size,
             "logits shape mismatch");

    DLTensor* log_probs = dl_log_softmax(logits, -1);
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int target = targets[b];
        DL_CHECK(target >= 0 && target < vocab_size, "target out of range");
        total_loss -= log_probs->data[b * vocab_size + target];
    }

    dl_tensor_free(log_probs);
    return dl_tensor_scalar(total_loss / batch_size);
}

/* ========== Embedding Lookup ========== */

DLTensor* dl_embedding_forward(DLTensor* weight, const int* indices, int n) {
    DL_CHECK(weight->ndim == 2, "embedding weight must be 2D");
    int embed_dim = weight->shape[1];
    int out_shape[2] = {n, embed_dim};
    DLTensor* out = dl_tensor_create(out_shape, 2);

    for (int i = 0; i < n; i++) {
        int idx = indices[i];
        DL_CHECK(idx >= 0 && idx < weight->shape[0], "embedding index out of range");
        memcpy(out->data + i * embed_dim, weight->data + idx * embed_dim,
               embed_dim * sizeof(float));
    }
    return out;
}

/* ========== Utility Operations ========== */

DLTensor* dl_concat(DLTensor** tensors, int n, int dim) {
    DL_CHECK(n > 0, "need at least one tensor to concat");
    if (dim < 0) dim += tensors[0]->ndim;

    int out_shape[DL_MAX_DIMS];
    memcpy(out_shape, tensors[0]->shape, tensors[0]->ndim * sizeof(int));
    out_shape[dim] = 0;
    for (int i = 0; i < n; i++) {
        out_shape[dim] += tensors[i]->shape[dim];
    }

    DLTensor* out = dl_tensor_create(out_shape, tensors[0]->ndim);

    int offset = 0;
    int outer = 1, inner = 1;
    for (int d = 0; d < dim; d++) outer *= out_shape[d];
    for (int d = dim + 1; d < tensors[0]->ndim; d++) inner *= out_shape[d];

    for (int t = 0; t < n; t++) {
        DLTensor* ct = dl_tensor_contiguous(tensors[t]);
        int t_dim = tensors[t]->shape[dim];
        for (int o = 0; o < outer; o++) {
            memcpy(out->data + (o * out_shape[dim] + offset) * inner,
                   ct->data + o * t_dim * inner,
                   t_dim * inner * sizeof(float));
        }
        offset += t_dim;
        dl_tensor_free(ct);
    }
    return out;
}

DLTensor* dl_split(DLTensor* t, int n_splits, int dim, int split_idx) {
    if (dim < 0) dim += t->ndim;
    DL_CHECK(t->shape[dim] % n_splits == 0, "split dim not divisible");
    int split_size = t->shape[dim] / n_splits;

    int out_shape[DL_MAX_DIMS];
    memcpy(out_shape, t->shape, t->ndim * sizeof(int));
    out_shape[dim] = split_size;

    DLTensor* out = dl_tensor_create(out_shape, t->ndim);
    DLTensor* ct = dl_tensor_contiguous(t);

    int outer = 1, inner = 1;
    for (int d = 0; d < dim; d++) outer *= t->shape[d];
    for (int d = dim + 1; d < t->ndim; d++) inner *= t->shape[d];

    for (int o = 0; o < outer; o++) {
        memcpy(out->data + o * split_size * inner,
               ct->data + (o * t->shape[dim] + split_idx * split_size) * inner,
               split_size * inner * sizeof(float));
    }
    dl_tensor_free(ct);
    return out;
}
