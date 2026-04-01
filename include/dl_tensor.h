#ifndef DL_TENSOR_H
#define DL_TENSOR_H

#include "dl_common.h"

/* Forward declarations */
typedef struct DLTensor DLTensor;
typedef struct DLGraphNode DLGraphNode;

/* Backward function type */
typedef void (*DLBackwardFn)(DLGraphNode* node);

/* Graph node for autograd */
struct DLGraphNode {
    DLTensor* output;
    DLTensor* inputs[4];
    int n_inputs;
    DLBackwardFn backward_fn;
    /* Extra data for backward (e.g., saved tensors, scalars) */
    void* extra;
    int extra_int[4];
    float extra_float[4];
    bool visited;
};

/* Tensor structure */
struct DLTensor {
    float* data;
    int shape[DL_MAX_DIMS];
    int strides[DL_MAX_DIMS];
    int ndim;
    int size;           /* total number of elements */

    bool requires_grad;
    DLTensor* grad;
    DLGraphNode* graph_node; /* backprop graph node */

    bool owns_data;     /* whether to free data on destroy */
    int ref_count;
};

/* Creation */
DLTensor* dl_tensor_create(const int* shape, int ndim);
DLTensor* dl_tensor_zeros(const int* shape, int ndim);
DLTensor* dl_tensor_ones(const int* shape, int ndim);
DLTensor* dl_tensor_rand(const int* shape, int ndim);
DLTensor* dl_tensor_randn(const int* shape, int ndim, float mean, float std);
DLTensor* dl_tensor_from_data(const float* data, const int* shape, int ndim);
DLTensor* dl_tensor_scalar(float value);
DLTensor* dl_tensor_clone(const DLTensor* src);

/* Memory management */
void dl_tensor_free(DLTensor* t);
DLTensor* dl_tensor_ref(DLTensor* t);

/* Shape operations (views - no copy) */
DLTensor* dl_tensor_reshape(DLTensor* t, const int* new_shape, int new_ndim);
DLTensor* dl_tensor_transpose(DLTensor* t, int dim0, int dim1);
DLTensor* dl_tensor_view(DLTensor* t, const int* new_shape, int new_ndim);

/* Contiguous copy */
DLTensor* dl_tensor_contiguous(DLTensor* t);
bool dl_tensor_is_contiguous(const DLTensor* t);

/* Element-wise operations */
DLTensor* dl_tensor_add(DLTensor* a, DLTensor* b);
DLTensor* dl_tensor_sub(DLTensor* a, DLTensor* b);
DLTensor* dl_tensor_mul(DLTensor* a, DLTensor* b);
DLTensor* dl_tensor_div(DLTensor* a, DLTensor* b);
DLTensor* dl_tensor_scale(DLTensor* a, float scalar);
DLTensor* dl_tensor_add_scalar(DLTensor* a, float scalar);

/* In-place operations */
void dl_tensor_add_(DLTensor* a, const DLTensor* b);
void dl_tensor_sub_(DLTensor* a, const DLTensor* b);
void dl_tensor_mul_(DLTensor* a, const DLTensor* b);
void dl_tensor_scale_(DLTensor* a, float scalar);
void dl_tensor_fill_(DLTensor* a, float value);
void dl_tensor_zero_(DLTensor* a);

/* Reduction */
DLTensor* dl_tensor_sum(DLTensor* t, int dim, bool keepdim);
DLTensor* dl_tensor_mean(DLTensor* t, int dim, bool keepdim);
DLTensor* dl_tensor_max(DLTensor* t, int dim, bool keepdim);
float dl_tensor_sum_all(const DLTensor* t);
float dl_tensor_max_all(const DLTensor* t);

/* Utility */
void dl_tensor_print(const DLTensor* t, const char* name);
int dl_tensor_flat_idx(const DLTensor* t, const int* indices);
float dl_tensor_get(const DLTensor* t, const int* indices);
void dl_tensor_set(DLTensor* t, const int* indices, float value);
bool dl_tensor_shape_eq(const DLTensor* a, const DLTensor* b);

/* SIMD-accelerated operations */
void dl_vec_add(float* out, const float* a, const float* b, int n);
void dl_vec_mul(float* out, const float* a, const float* b, int n);
void dl_vec_scale(float* out, const float* a, float s, int n);
float dl_vec_dot(const float* a, const float* b, int n);
void dl_vec_copy(float* dst, const float* src, int n);

/* Gradient management */
void dl_tensor_set_requires_grad(DLTensor* t, bool requires);
void dl_tensor_zero_grad(DLTensor* t);
DLTensor* dl_tensor_ensure_grad(DLTensor* t);

#endif /* DL_TENSOR_H */
