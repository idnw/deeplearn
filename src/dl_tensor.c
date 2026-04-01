#include "dl_tensor.h"

/* ========== SIMD Vector Operations ========== */

void dl_vec_add(float* out, const float* a, const float* b, int n) {
    int i = 0;
#if DL_USE_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#elif DL_USE_SSE2
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_add_ps(va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];
}

void dl_vec_mul(float* out, const float* a, const float* b, int n) {
    int i = 0;
#if DL_USE_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#elif DL_USE_SSE2
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * b[i];
}

void dl_vec_scale(float* out, const float* a, float s, int n) {
    int i = 0;
#if DL_USE_AVX2
    __m256 vs = _mm256_set1_ps(s);
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vs));
    }
#elif DL_USE_SSE2
    __m128 vs = _mm_set1_ps(s);
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vs));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * s;
}

float dl_vec_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    int i = 0;
#if DL_USE_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
    }
    float buf[8];
    _mm256_storeu_ps(buf, vsum);
    for (int j = 0; j < 8; j++) sum += buf[j];
#elif DL_USE_SSE2
    __m128 vsum = _mm_setzero_ps();
    for (; i + 3 < n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
    }
    float buf[4];
    _mm_storeu_ps(buf, vsum);
    for (int j = 0; j < 4; j++) sum += buf[j];
#endif
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void dl_vec_copy(float* dst, const float* src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========== Tensor Creation ========== */

static void dl_tensor_compute_strides(DLTensor* t) {
    if (t->ndim == 0) return;
    t->strides[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
}

static int dl_tensor_compute_size(const int* shape, int ndim) {
    if (ndim == 0) return 1;
    int size = 1;
    for (int i = 0; i < ndim; i++) size *= shape[i];
    return size;
}

DLTensor* dl_tensor_create(const int* shape, int ndim) {
    DL_CHECK(ndim >= 0 && ndim <= DL_MAX_DIMS, "invalid ndim");
    DLTensor* t = DL_ALLOC(DLTensor, 1);
    memset(t, 0, sizeof(DLTensor));
    t->ndim = ndim;
    t->size = dl_tensor_compute_size(shape, ndim);
    memcpy(t->shape, shape, ndim * sizeof(int));
    dl_tensor_compute_strides(t);
    t->data = (float*)dl_calloc(t->size > 0 ? t->size : 1, sizeof(float));
    t->owns_data = true;
    t->ref_count = 1;
    return t;
}


DLTensor* dl_tensor_zeros(const int* shape, int ndim) {
    return dl_tensor_create(shape, ndim);
}

DLTensor* dl_tensor_ones(const int* shape, int ndim) {
    DLTensor* t = dl_tensor_create(shape, ndim);
    for (int i = 0; i < t->size; i++) t->data[i] = 1.0f;
    return t;
}

DLTensor* dl_tensor_rand(const int* shape, int ndim) {
    dl_rng_ensure_init();
    DLTensor* t = dl_tensor_create(shape, ndim);
    for (int i = 0; i < t->size; i++) t->data[i] = dl_rng_float(&dl_global_rng);
    return t;
}

DLTensor* dl_tensor_randn(const int* shape, int ndim, float mean, float std) {
    dl_rng_ensure_init();
    DLTensor* t = dl_tensor_create(shape, ndim);
    for (int i = 0; i < t->size; i++) {
        t->data[i] = mean + std * dl_rng_normal(&dl_global_rng);
    }
    return t;
}

DLTensor* dl_tensor_from_data(const float* data, const int* shape, int ndim) {
    DLTensor* t = dl_tensor_create(shape, ndim);
    memcpy(t->data, data, t->size * sizeof(float));
    return t;
}

DLTensor* dl_tensor_scalar(float value) {
    DLTensor* t = dl_tensor_create(NULL, 0);
    t->data[0] = value;
    return t;
}

DLTensor* dl_tensor_clone(const DLTensor* src) {
    DLTensor* dst = dl_tensor_create(src->shape, src->ndim);
    if (dl_tensor_is_contiguous(src)) {
        memcpy(dst->data, src->data, src->size * sizeof(float));
    } else {
        /* Copy element by element for non-contiguous tensors */
        int indices[DL_MAX_DIMS] = {0};
        for (int i = 0; i < src->size; i++) {
            float val = 0.0f;
            int offset = 0;
            for (int d = 0; d < src->ndim; d++) {
                offset += indices[d] * src->strides[d];
            }
            val = src->data[offset];
            dst->data[i] = val;

            /* Increment indices */
            for (int d = src->ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < src->shape[d]) break;
                indices[d] = 0;
            }
        }
    }
    return dst;
}

/* ========== Memory Management ========== */

void dl_tensor_free(DLTensor* t) {
    if (!t) return;
    t->ref_count--;
    if (t->ref_count <= 0) {
        if (t->owns_data && t->data) free(t->data);
        free(t);
    }
}

DLTensor* dl_tensor_ref(DLTensor* t) {
    if (t) t->ref_count++;
    return t;
}

/* ========== Shape Operations ========== */

DLTensor* dl_tensor_reshape(DLTensor* t, const int* new_shape, int new_ndim) {
    int new_size = dl_tensor_compute_size(new_shape, new_ndim);
    DL_CHECK(new_size == t->size, "reshape size mismatch");

    if (dl_tensor_is_contiguous(t)) {
        DLTensor* out = DL_ALLOC(DLTensor, 1);
        memset(out, 0, sizeof(DLTensor));
        out->data = t->data;
        out->ndim = new_ndim;
        out->size = new_size;
        memcpy(out->shape, new_shape, new_ndim * sizeof(int));
        dl_tensor_compute_strides(out);
        out->owns_data = false;
        out->ref_count = 1;
        dl_tensor_ref(t); /* keep source alive */
        return out;
    } else {
        DLTensor* contig = dl_tensor_contiguous(t);
        DLTensor* out = dl_tensor_reshape(contig, new_shape, new_ndim);
        out->owns_data = true;
        contig->owns_data = false;
        dl_tensor_free(contig);
        return out;
    }
}

DLTensor* dl_tensor_transpose(DLTensor* t, int dim0, int dim1) {
    DL_CHECK(dim0 >= 0 && dim0 < t->ndim, "invalid dim0");
    DL_CHECK(dim1 >= 0 && dim1 < t->ndim, "invalid dim1");

    DLTensor* out = DL_ALLOC(DLTensor, 1);
    memset(out, 0, sizeof(DLTensor));
    out->data = t->data;
    out->ndim = t->ndim;
    out->size = t->size;
    memcpy(out->shape, t->shape, t->ndim * sizeof(int));
    memcpy(out->strides, t->strides, t->ndim * sizeof(int));

    /* Swap dimensions */
    int tmp = out->shape[dim0];
    out->shape[dim0] = out->shape[dim1];
    out->shape[dim1] = tmp;
    tmp = out->strides[dim0];
    out->strides[dim0] = out->strides[dim1];
    out->strides[dim1] = tmp;

    out->owns_data = false;
    out->ref_count = 1;
    dl_tensor_ref(t);
    return out;
}

DLTensor* dl_tensor_view(DLTensor* t, const int* new_shape, int new_ndim) {
    return dl_tensor_reshape(t, new_shape, new_ndim);
}

bool dl_tensor_is_contiguous(const DLTensor* t) {
    if (t->ndim == 0) return true;
    int expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

DLTensor* dl_tensor_contiguous(DLTensor* t) {
    if (dl_tensor_is_contiguous(t)) {
        return dl_tensor_ref(t);
    }
    return dl_tensor_clone(t);
}

/* ========== Element-wise Operations ========== */

/* Broadcast: compute output shape */
static bool dl_broadcast_shape(const DLTensor* a, const DLTensor* b,
                                int* out_shape, int* out_ndim) {
    *out_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < *out_ndim; i++) {
        int da = i < *out_ndim - a->ndim ? 1 : a->shape[i - (*out_ndim - a->ndim)];
        int db = i < *out_ndim - b->ndim ? 1 : b->shape[i - (*out_ndim - b->ndim)];
        if (da != db && da != 1 && db != 1) return false;
        out_shape[i] = da > db ? da : db;
    }
    return true;
}

static int dl_broadcast_offset(const DLTensor* t, const int* indices, int ndim) {
    int offset = 0;
    int dim_offset = ndim - t->ndim;
    for (int d = 0; d < t->ndim; d++) {
        int idx = indices[d + dim_offset];
        if (t->shape[d] == 1) idx = 0; /* broadcast */
        offset += idx * t->strides[d];
    }
    return offset;
}

typedef float (*dl_binary_op)(float, float);

static float dl_op_add(float a, float b) { return a + b; }
static float dl_op_sub(float a, float b) { return a - b; }
static float dl_op_mul(float a, float b) { return a * b; }
static float dl_op_div(float a, float b) { return a / b; }

static DLTensor* dl_tensor_binary_op(DLTensor* a, DLTensor* b, dl_binary_op op) {
    int out_shape[DL_MAX_DIMS];
    int out_ndim;
    DL_CHECK(dl_broadcast_shape(a, b, out_shape, &out_ndim), "broadcast shape mismatch");

    DLTensor* out = dl_tensor_create(out_shape, out_ndim);

    /* Fast path: same shape, contiguous */
    if (dl_tensor_shape_eq(a, b) && dl_tensor_is_contiguous(a) && dl_tensor_is_contiguous(b)) {
        if (op == dl_op_add) {
            dl_vec_add(out->data, a->data, b->data, out->size);
        } else if (op == dl_op_mul) {
            dl_vec_mul(out->data, a->data, b->data, out->size);
        } else {
            for (int i = 0; i < out->size; i++)
                out->data[i] = op(a->data[i], b->data[i]);
        }
        return out;
    }

    /* General broadcast path */
    int indices[DL_MAX_DIMS] = {0};
    for (int i = 0; i < out->size; i++) {
        float va = a->data[dl_broadcast_offset(a, indices, out_ndim)];
        float vb = b->data[dl_broadcast_offset(b, indices, out_ndim)];
        out->data[i] = op(va, vb);

        for (int d = out_ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < out_shape[d]) break;
            indices[d] = 0;
        }
    }
    return out;
}

DLTensor* dl_tensor_add(DLTensor* a, DLTensor* b) { return dl_tensor_binary_op(a, b, dl_op_add); }
DLTensor* dl_tensor_sub(DLTensor* a, DLTensor* b) { return dl_tensor_binary_op(a, b, dl_op_sub); }
DLTensor* dl_tensor_mul(DLTensor* a, DLTensor* b) { return dl_tensor_binary_op(a, b, dl_op_mul); }
DLTensor* dl_tensor_div(DLTensor* a, DLTensor* b) { return dl_tensor_binary_op(a, b, dl_op_div); }

DLTensor* dl_tensor_scale(DLTensor* a, float scalar) {
    DLTensor* out = dl_tensor_create(a->shape, a->ndim);
    if (dl_tensor_is_contiguous(a)) {
        dl_vec_scale(out->data, a->data, scalar, a->size);
    } else {
        DLTensor* c = dl_tensor_contiguous(a);
        dl_vec_scale(out->data, c->data, scalar, c->size);
        dl_tensor_free(c);
    }
    return out;
}

DLTensor* dl_tensor_add_scalar(DLTensor* a, float scalar) {
    DLTensor* out = dl_tensor_clone(a);
    for (int i = 0; i < out->size; i++) out->data[i] += scalar;
    return out;
}

/* ========== In-place Operations ========== */

void dl_tensor_add_(DLTensor* a, const DLTensor* b) {
    DL_CHECK(dl_tensor_shape_eq(a, b), "shape mismatch for in-place add");
    dl_vec_add(a->data, a->data, b->data, a->size);
}

void dl_tensor_sub_(DLTensor* a, const DLTensor* b) {
    DL_CHECK(dl_tensor_shape_eq(a, b), "shape mismatch for in-place sub");
    for (int i = 0; i < a->size; i++) a->data[i] -= b->data[i];
}

void dl_tensor_mul_(DLTensor* a, const DLTensor* b) {
    DL_CHECK(dl_tensor_shape_eq(a, b), "shape mismatch for in-place mul");
    dl_vec_mul(a->data, a->data, b->data, a->size);
}

void dl_tensor_scale_(DLTensor* a, float scalar) {
    dl_vec_scale(a->data, a->data, scalar, a->size);
}

void dl_tensor_fill_(DLTensor* a, float value) {
    for (int i = 0; i < a->size; i++) a->data[i] = value;
}

void dl_tensor_zero_(DLTensor* a) {
    memset(a->data, 0, a->size * sizeof(float));
}

/* ========== Reductions ========== */

DLTensor* dl_tensor_sum(DLTensor* t, int dim, bool keepdim) {
    DL_CHECK(dim >= 0 && dim < t->ndim, "invalid reduction dim");

    int out_shape[DL_MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < t->ndim; d++) {
        if (d == dim) {
            if (keepdim) out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t->shape[d];
        }
    }

    DLTensor* out = dl_tensor_zeros(out_shape, out_ndim);

    int indices[DL_MAX_DIMS] = {0};
    for (int i = 0; i < t->size; i++) {
        int src_offset = 0;
        for (int d = 0; d < t->ndim; d++)
            src_offset += indices[d] * t->strides[d];

        /* Compute output index */
        int out_indices[DL_MAX_DIMS];
        int oi = 0;
        for (int d = 0; d < t->ndim; d++) {
            if (d == dim) {
                if (keepdim) out_indices[oi++] = 0;
            } else {
                out_indices[oi++] = indices[d];
            }
        }
        int dst_offset = 0;
        for (int d = 0; d < out_ndim; d++)
            dst_offset += out_indices[d] * out->strides[d];

        out->data[dst_offset] += t->data[src_offset];

        for (int d = t->ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < t->shape[d]) break;
            indices[d] = 0;
        }
    }
    return out;
}

DLTensor* dl_tensor_mean(DLTensor* t, int dim, bool keepdim) {
    DLTensor* s = dl_tensor_sum(t, dim, keepdim);
    dl_tensor_scale_(s, 1.0f / t->shape[dim]);
    return s;
}

DLTensor* dl_tensor_max(DLTensor* t, int dim, bool keepdim) {
    DL_CHECK(dim >= 0 && dim < t->ndim, "invalid reduction dim");

    int out_shape[DL_MAX_DIMS];
    int out_ndim = 0;
    for (int d = 0; d < t->ndim; d++) {
        if (d == dim) {
            if (keepdim) out_shape[out_ndim++] = 1;
        } else {
            out_shape[out_ndim++] = t->shape[d];
        }
    }

    DLTensor* out = dl_tensor_create(out_shape, out_ndim);
    dl_tensor_fill_(out, -FLT_MAX);

    int indices[DL_MAX_DIMS] = {0};
    for (int i = 0; i < t->size; i++) {
        int src_offset = 0;
        for (int d = 0; d < t->ndim; d++)
            src_offset += indices[d] * t->strides[d];

        int out_indices[DL_MAX_DIMS];
        int oi = 0;
        for (int d = 0; d < t->ndim; d++) {
            if (d == dim) {
                if (keepdim) out_indices[oi++] = 0;
            } else {
                out_indices[oi++] = indices[d];
            }
        }
        int dst_offset = 0;
        for (int d = 0; d < out_ndim; d++)
            dst_offset += out_indices[d] * out->strides[d];

        if (t->data[src_offset] > out->data[dst_offset])
            out->data[dst_offset] = t->data[src_offset];

        for (int d = t->ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < t->shape[d]) break;
            indices[d] = 0;
        }
    }
    return out;
}

float dl_tensor_sum_all(const DLTensor* t) {
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) sum += t->data[i];
    return sum;
}

float dl_tensor_max_all(const DLTensor* t) {
    float m = -FLT_MAX;
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] > m) m = t->data[i];
    }
    return m;
}

/* ========== Utility ========== */

void dl_tensor_print(const DLTensor* t, const char* name) {
    printf("%s: shape=[", name ? name : "tensor");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i < t->ndim - 1 ? "," : "");
    }
    printf("], data=[");
    int n = t->size < 10 ? t->size : 10;
    for (int i = 0; i < n; i++) {
        printf("%.4f%s", t->data[i], i < n - 1 ? ", " : "");
    }
    if (t->size > 10) printf(", ...");
    printf("]\n");
}

int dl_tensor_flat_idx(const DLTensor* t, const int* indices) {
    int offset = 0;
    for (int d = 0; d < t->ndim; d++)
        offset += indices[d] * t->strides[d];
    return offset;
}

float dl_tensor_get(const DLTensor* t, const int* indices) {
    return t->data[dl_tensor_flat_idx(t, indices)];
}

void dl_tensor_set(DLTensor* t, const int* indices, float value) {
    t->data[dl_tensor_flat_idx(t, indices)] = value;
}

bool dl_tensor_shape_eq(const DLTensor* a, const DLTensor* b) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

/* ========== Gradient Management ========== */

void dl_tensor_set_requires_grad(DLTensor* t, bool requires) {
    t->requires_grad = requires;
    if (requires && !t->grad) {
        t->grad = dl_tensor_zeros(t->shape, t->ndim);
    }
}

void dl_tensor_zero_grad(DLTensor* t) {
    if (t->grad) dl_tensor_zero_(t->grad);
}

/* Forward declaration - defined in dl_autograd.c */
void dl_track_grad_data(float* data);

DLTensor* dl_tensor_ensure_grad(DLTensor* t) {
    if (!t->grad) {
        t->grad = dl_tensor_zeros(t->shape, t->ndim);
        /* If this is an intermediate tensor (not a parameter), track grad data */
        if (!t->requires_grad) {
            dl_track_grad_data(t->grad->data);
            t->grad->owns_data = false;
        }
    }
    return t->grad;
}
