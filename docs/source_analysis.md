# DeepLearn-C 源码详细分析

## 目录

1. [架构总览](#1-架构总览)
2. [dl_common.h — 基础设施层](#2-dl_commonh--基础设施层)
3. [dl_tensor — 张量引擎](#3-dl_tensor--张量引擎)
4. [dl_ops — 数学运算库](#4-dl_ops--数学运算库)
5. [dl_autograd — 自动微分引擎](#5-dl_autograd--自动微分引擎)
6. [dl_nn — 神经网络层](#6-dl_nn--神经网络层)
7. [dl_transformer — Transformer 架构](#7-dl_transformer--transformer-架构)
8. [dl_optimizer — 优化器](#8-dl_optimizer--优化器)
9. [dl_tokenizer — 分词器](#9-dl_tokenizer--分词器)
10. [dl_dataloader — 数据加载器](#10-dl_dataloader--数据加载器)
11. [dl_serialize — 模型序列化](#11-dl_serialize--模型序列化)
12. [内存管理机制详解](#12-内存管理机制详解)
13. [计算图与反向传播详解](#13-计算图与反向传播详解)
14. [SIMD 加速实现](#14-simd-加速实现)

---

## 1. 架构总览

### 1.1 模块依赖图

```
                  dl_common.h
                  (宏, 内存, RNG, SIMD检测)
                       |
                  dl_tensor.h/c
                  (N维张量, SIMD向量运算, 引用计数)
                       |
                  dl_ops.h/c
                  (矩阵乘, softmax, 激活, 损失)
                       |
                  dl_autograd.h/c
                  (计算图, 反向传播, 15种可微分运算)
                       |
                  dl_nn.h/c
                  (Linear, Embedding, LayerNorm, Dropout)
                       |
               dl_transformer.h/c ----+---- dl_optimizer.h/c
               (MHA, Block, GPT模型)  |    (SGD, Adam, LR调度)
                       |               |
               dl_serialize.h/c -------+
               (自定义格式, GGUF)

    dl_tokenizer.h/c ----> dl_dataloader.h/c
    (字符级, BPE)         (批处理, shuffle)
```

### 1.2 设计原则

| 原则 | 实现方式 |
|------|----------|
| 零外部依赖 | 仅使用 C99 标准库 + POSIX clock_gettime |
| 行优先存储 | 张量以 row-major 存储，strides 自动计算 |
| 引用计数 | `ref_count` 管理共享数据的张量生命周期 |
| 分离关注点 | 原始运算(dl_ops) 和 可微分运算(dl_ag_*) 分开 |
| 编译期 SIMD | 通过预处理宏 `__AVX2__` / `__SSE2__` 自动选择 |

### 1.3 文件清单

```
include/
  dl_common.h        148 行  基础设施（内存, RNG, SIMD, 计时器）
  dl_tensor.h        117 行  张量数据结构和操作声明
  dl_ops.h            35 行  数学运算声明
  dl_autograd.h       58 行  自动微分引擎声明
  dl_nn.h             85 行  神经网络层声明
  dl_transformer.h    78 行  Transformer 架构声明
  dl_optimizer.h      53 行  优化器声明
  dl_tokenizer.h      30 行  分词器声明
  dl_dataloader.h     30 行  数据加载器声明
  dl_serialize.h      62 行  序列化声明

src/
  dl_tensor.c        453 行  张量运算实现
  dl_ops.c           309 行  数学运算实现
  dl_autograd.c      583 行  自动微分实现
  dl_nn.c            168 行  NN层实现
  dl_transformer.c   380 行  Transformer实现
  dl_optimizer.c     185 行  优化器实现
  dl_tokenizer.c     173 行  分词器实现
  dl_dataloader.c     86 行  数据加载器实现
  dl_serialize.c     279 行  序列化实现
```

---

## 2. dl_common.h — 基础设施层

这是最底层的头文件，被所有其他模块包含。

### 2.1 SIMD 检测

```c
#ifdef __AVX2__
    #include <immintrin.h>
    #define DL_USE_AVX2 1      // AVX2: 256位, 8个float并行
#elif defined(__SSE2__)
    #include <emmintrin.h>
    #define DL_USE_SSE2 1      // SSE2: 128位, 4个float并行
#endif
```

编译器根据 `-mavx2` 或 `-msse2` 标志定义这些宏。框架代码中使用 `#if DL_USE_AVX2` 选择对应的 SIMD 路径。

### 2.2 内存工具

```c
// 安全分配（失败时终止程序）
static inline void* dl_malloc(size_t size);
static inline void* dl_calloc(size_t count, size_t size);
static inline void* dl_realloc(void* ptr, size_t size);

// 便捷宏
#define DL_ALLOC(type, count)  ((type*)dl_malloc(sizeof(type) * (count)))
#define DL_FREE(ptr)           do { free(ptr); (ptr) = NULL; } while(0)
```

所有内存分配都通过这些包装函数，确保 OOM 时立即报错而非返回 NULL。

### 2.3 Arena 分配器

```c
typedef struct {
    char* base;       // 预分配的连续内存块
    size_t used;      // 当前已用字节数
    size_t capacity;  // 总容量
} DLArena;
```

Arena 是一种批量分配/释放模式：分配时仅移动指针（O(1)），释放时重置指针（整块释放）。所有分配按 **32 字节对齐** 以满足 AVX2 要求。

**用途：** 计算图的中间张量数据使用 arena 管理。

### 2.4 随机数生成器

```c
typedef struct {
    uint64_t s[4];    // 256位内部状态
} DLRng;
```

使用 **xoshiro256**** 算法——速度快且统计质量好：
- `dl_rng_next()` — 生成 64 位随机整数（核心操作：旋转、移位、异或）
- `dl_rng_float()` — 转换为 `[0, 1)` 均匀分布浮点数
- `dl_rng_normal()` — **Box-Muller 变换** 生成标准正态分布

种子初始化使用 **SplitMix64** 算法扩展单个 seed 为 4 个状态字。

### 2.5 时间测量

```c
static inline double dl_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
```

使用 POSIX `clock_gettime(CLOCK_MONOTONIC)` 获取单调时钟，精度到纳秒级。

---

## 3. dl_tensor — 张量引擎

### 3.1 核心数据结构

```c
struct DLTensor {
    float* data;                 // 指向浮点数据的指针
    int shape[DL_MAX_DIMS];      // 各维度大小 (最多8维)
    int strides[DL_MAX_DIMS];    // 各维度步长 (元素为单位)
    int ndim;                    // 维度数
    int size;                    // 元素总数 (所有shape之积)

    bool requires_grad;          // 是否需要计算梯度
    DLTensor* grad;              // 梯度张量 (与自身同形状)
    DLGraphNode* graph_node;     // 计算图节点 (反向传播用)

    bool owns_data;              // 是否拥有 data 的所有权
    int ref_count;               // 引用计数
};
```

**Strides 机制：** 每个维度的 stride 表示沿该维度移动一个位置需要跳过多少元素。对于行优先的 `(3, 4)` 张量：
```
shape   = [3, 4]
strides = [4, 1]    // 行步长=4, 列步长=1
```

### 3.2 Stride 计算

```c
static void dl_tensor_compute_strides(DLTensor* t) {
    t->strides[t->ndim - 1] = 1;                           // 最内层步长=1
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1]; // 外层=内层*内层形状
    }
}
```

### 3.3 视图与连续性

**转置** 不拷贝数据，只交换 shape 和 strides：
```c
DLTensor* dl_tensor_transpose(DLTensor* t, int dim0, int dim1) {
    // ... 创建新 DLTensor，共享 t->data ...
    // 交换 shape[dim0] <-> shape[dim1]
    // 交换 strides[dim0] <-> strides[dim1]
    out->owns_data = false;   // 不拥有数据
    dl_tensor_ref(t);          // 增加原张量引用计数
    return out;
}
```

**连续性检查** — 验证 strides 是否与行优先连续存储一致：
```c
bool dl_tensor_is_contiguous(const DLTensor* t) {
    int expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}
```

### 3.4 广播机制

二元运算（加/减/乘/除）支持 NumPy 风格的广播。核心函数：

```c
static bool dl_broadcast_shape(const DLTensor* a, const DLTensor* b,
                                int* out_shape, int* out_ndim) {
    *out_ndim = max(a->ndim, b->ndim);
    for (int i = 0; i < *out_ndim; i++) {
        int da = (对应a的维度大小，不存在时为1);
        int db = (对应b的维度大小，不存在时为1);
        if (da != db && da != 1 && db != 1) return false;  // 不兼容
        out_shape[i] = max(da, db);
    }
    return true;
}
```

广播访问通过 `dl_broadcast_offset` 实现——维度大小为 1 的维度上索引始终为 0。

**快速路径优化：** 当两个张量形状完全相同且连续时，直接调用 SIMD 向量运算：
```c
if (dl_tensor_shape_eq(a, b) && dl_tensor_is_contiguous(a) && dl_tensor_is_contiguous(b)) {
    dl_vec_add(out->data, a->data, b->data, out->size);  // SIMD
} else {
    // 通用广播路径（逐元素）
}
```

### 3.5 SIMD 向量运算

以向量加法为例：

```c
void dl_vec_add(float* out, const float* a, const float* b, int n) {
    int i = 0;
#if DL_USE_AVX2
    for (; i + 7 < n; i += 8) {                           // 每次处理8个float
        __m256 va = _mm256_loadu_ps(a + i);                // 加载256位(8个float)
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));  // 8路并行加法
    }
#elif DL_USE_SSE2
    for (; i + 3 < n; i += 4) {                           // 每次处理4个float
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_add_ps(va, vb));
    }
#endif
    for (; i < n; i++) out[i] = a[i] + b[i];              // 标量处理剩余元素
}
```

使用 `_loadu_ps`（非对齐加载）而非 `_load_ps`（对齐加载）以处理任意对齐的内存。

### 3.6 引用计数与释放

```c
void dl_tensor_free(DLTensor* t) {
    if (!t) return;
    t->ref_count--;
    if (t->ref_count <= 0) {
        if (t->owns_data && t->data) free(t->data);  // 只有拥有者释放数据
        free(t);
    }
}
```

- `dl_tensor_ref(t)` 增加引用计数
- 视图（reshape/transpose）将 `owns_data = false` 并增加源张量引用
- 只有 `owns_data == true` 的张量在释放时才释放数据内存

---

## 4. dl_ops — 数学运算库

### 4.1 分块矩阵乘法

```c
#define TILE_SIZE 32

static void dl_matmul_tiled(float* C, const float* A, const float* B,
                             int M, int K, int N) {
    memset(C, 0, M * N * sizeof(float));
    for (int i0 = 0; i0 < M; i0 += TILE_SIZE) {        // 外层：行分块
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {    // 外层：列分块
            for (int k0 = 0; k0 < K; k0 += TILE_SIZE) { // 外层：内维分块
                // 内层：处理 TILE_SIZE x TILE_SIZE 的子矩阵
                for (int i = i0; i < min(i0+TILE, M); i++) {
                    for (int k = k0; k < min(k0+TILE, K); k++) {
                        float a_ik = A[i * K + k];
                        // SIMD 处理 j 维度
                        __m256 va = _mm256_set1_ps(a_ik);
                        for (j = j0; j + 7 < jmax; j += 8) {
                            // C[i,j:j+8] += a_ik * B[k,j:j+8]
                        }
                    }
                }
            }
        }
    }
}
```

**为什么用分块？** CPU 缓存（L1 约 32KB）只能容纳有限数据。32x32 的 float 块 = 4KB，三个块 12KB 可以完全放入 L1 缓存，避免频繁的缓存未命中。

### 4.2 数值稳定的 Softmax

```c
DLTensor* dl_softmax(DLTensor* t, int dim) {
    // 1. 找到每行最大值
    float max_val = -FLT_MAX;
    for (...) if (data[idx] > max_val) max_val = data[idx];

    // 2. 减去最大值后求 exp（防止溢出）
    float sum = 0;
    for (...) {
        data[idx] = expf(data[idx] - max_val);   // exp(x - max) <= 1，不会溢出
        sum += data[idx];
    }

    // 3. 归一化
    for (...) data[idx] /= sum;
}
```

**关键：** `exp(x)` 当 x > 88 时溢出为 inf。减去 max 后 `x - max <= 0`，`exp(x - max) <= 1`。

### 4.3 层归一化

```c
DLTensor* dl_layer_norm(DLTensor* x, DLTensor* gamma, DLTensor* beta, float eps) {
    for (int b = 0; b < batch_size; b++) {
        // 1. 计算均值 μ
        float mean = sum(row) / norm_size;

        // 2. 计算方差 σ²
        float var = sum((row[i] - mean)²) / norm_size;

        // 3. 归一化: x̂ = (x - μ) / √(σ² + ε)
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < norm_size; i++) {
            float normalized = (row[i] - mean) * inv_std;
            out[i] = gamma[i] * normalized + beta[i];    // 仿射变换
        }
    }
}
```

### 4.4 GELU 激活函数

GPT-2 使用的激活函数，近似实现：

```c
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
float x3 = x * x * x;
float inner = 0.7978845608f * (x + 0.044715f * x3);  // √(2/π) ≈ 0.7979
result = 0.5f * x * (1.0f + tanhf(inner));
```

### 4.5 交叉熵损失

```c
DLTensor* dl_cross_entropy_loss(DLTensor* logits, const int* targets,
                                 int batch_size, int vocab_size) {
    // 1. 对 logits 做 log_softmax
    DLTensor* log_probs = dl_log_softmax(logits, -1);

    // 2. 取目标位置的负对数概率
    float total_loss = 0;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= log_probs->data[b * vocab_size + targets[b]];
    }

    // 3. 平均
    return dl_tensor_scalar(total_loss / batch_size);
}
```

数学公式：`L = -1/N * Σ log P(target_i)`

---

## 5. dl_autograd — 自动微分引擎

### 5.1 核心概念

自动微分使用 **反向模式**（reverse-mode AD），也叫反向传播。核心是维护一个 **有向无环图**（DAG），其中：
- **节点** = 运算（matmul, add, gelu, ...）
- **边** = 张量（输入/输出）

### 5.2 计算图结构

```c
typedef struct DLGraphNode {
    DLTensor* output;          // 该运算的输出张量
    DLTensor* inputs[4];       // 输入张量（最多4个）
    int n_inputs;
    DLBackwardFn backward_fn;  // 反向传播函数指针
    void* extra;               // 额外数据（如saved tensors）
    int extra_int[4];          // 整数参数（如维度索引）
    float extra_float[4];      // 浮点参数（如缩放系数）
    bool visited;              // 拓扑排序标记
} DLGraphNode;
```

全局图：
```c
typedef struct {
    DLGraphNode* nodes[262144];    // 所有节点
    int n_nodes;
    bool no_grad;                   // 是否禁用梯度

    float* tracked_data[786432];    // 中间张量数据指针（用于批量释放）
    int n_tracked;
} DLGraph;
```

### 5.3 节点注册

每个 `dl_ag_*` 函数：
1. 执行前向运算（调用 `dl_ops` 中的函数）
2. 创建 `DLGraphNode` 并注册到图中
3. 跟踪输出张量数据

```c
DLTensor* dl_ag_matmul(DLTensor* a, DLTensor* b) {
    DLTensor* out = dl_matmul(a, b);                     // 1. 前向
    DLTensor* inputs[] = {a, b};
    dl_graph_add_node(out, inputs, 2, backward_matmul);   // 2. 注册节点
    dl_graph_track(out);                                   // 3. 跟踪数据
    return out;
}
```

`dl_graph_add_node` 仅在输入中存在需要梯度的张量时才创建节点：
```c
bool needs_grad = false;
for (int i = 0; i < n_inputs; i++) {
    if (inputs[i] && (inputs[i]->requires_grad || inputs[i]->graph_node)) {
        needs_grad = true;
        break;
    }
}
if (!needs_grad) return NULL;  // 不需要梯度，不记录
```

### 5.4 反向传播

```c
void dl_backward(DLTensor* loss) {
    // 1. 设置初始梯度 dL/dL = 1
    loss->grad = dl_tensor_ones(loss->shape, loss->ndim);

    // 2. 拓扑排序（DFS后序遍历）
    DLGraphNode* sorted[MAX];
    int count = 0;
    dl_topo_sort(loss->graph_node, sorted, &count);

    // 3. 逆序遍历，执行每个节点的 backward 函数
    for (int i = count - 1; i >= 0; i--) {
        if (sorted[i]->backward_fn && sorted[i]->output->grad) {
            sorted[i]->backward_fn(sorted[i]);
        }
    }
}
```

### 5.5 各运算的反向函数

#### 矩阵乘法反向

对于 `C = A @ B`，梯度为：
```
dL/dA = dL/dC @ B^T
dL/dB = A^T @ dL/dC
```

```c
static void backward_matmul(DLGraphNode* node) {
    DLTensor* a = node->inputs[0];
    DLTensor* b = node->inputs[1];
    DLTensor* grad_out = node->output->grad;

    // dL/dA = grad_out @ B^T
    DLTensor* bt = dl_tensor_transpose(b, -2, -1);
    DLTensor* grad_a = dl_matmul(grad_out, bt);
    dl_accumulate_grad(a->grad, grad_a);   // 支持批次维度归约

    // dL/dB = A^T @ grad_out
    DLTensor* at = dl_tensor_transpose(a, -2, -1);
    DLTensor* grad_b = dl_matmul(at, grad_out);
    dl_accumulate_grad(b->grad, grad_b);   // 支持批次维度归约
}
```

`dl_accumulate_grad` 处理形状不匹配（批次维度归约）：
```c
static void dl_accumulate_grad(DLTensor* grad_param, DLTensor* grad_computed) {
    if (shape_eq) {
        dl_tensor_add_(grad_param, grad_computed);  // 同形状直接加
    } else {
        // 沿批次维度求和后累加
        int batch_size = grad_computed->size / grad_param->size;
        for (int b = 0; b < batch_size; b++)
            for (int i = 0; i < grad_param->size; i++)
                grad_param->data[i] += grad_computed->data[b * grad_param->size + i];
    }
}
```

#### GELU 反向

GELU(x) = 0.5x(1 + tanh(φ)), 其中 φ = √(2/π)(x + 0.044715x³)

```c
// dGELU/dx = 0.5(1 + tanh(φ)) + 0.5·x·sech²(φ)·dφ/dx
// dφ/dx = √(2/π)(1 + 3·0.044715·x²)
float tanh_val = tanhf(inner);
float sech2 = 1.0f - tanh_val * tanh_val;
float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
float grad = 0.5f * (1.0f + tanh_val) + 0.5f * xi * sech2 * d_inner;
```

#### Softmax 反向

对于 `y = softmax(x)`，雅可比矩阵为 `diag(y) - y·y^T`：

```c
// 简化公式：dx_i = y_i * (dL/dy_i - Σ_j dL/dy_j * y_j)
float sum = 0;
for (int a = 0; a < axis_size; a++)
    sum += grad_out[idx] * out[idx];  // Σ dL/dy_j * y_j

for (int a = 0; a < axis_size; a++)
    gx[idx] += out[idx] * (grad_out[idx] - sum);
```

#### 层归一化反向

层归一化反向是最复杂的反向传播之一：

```c
// 对 gamma 的梯度: dL/dγ = Σ_batch (dL/dy * x̂)    其中 x̂ 是归一化后的值
// 对 beta 的梯度:  dL/dβ = Σ_batch dL/dy
// 对输入的梯度:    dx = (1/σ) * (dy' - mean(dy') - x̂ * mean(dy' * x̂))
//                  其中 dy' = dL/dy * gamma
```

#### 交叉熵反向

```c
// dL/d(logits) = softmax(logits) - one_hot(target)
DLTensor* probs = dl_softmax(logits, -1);
for (int b = 0; b < batch; b++) {
    for (int v = 0; v < vocab; v++) {
        float grad = probs[b * vocab + v];
        if (v == targets[b]) grad -= 1.0f;   // 减去 one-hot
        gl[b * vocab + v] += grad * scale;
    }
}
```

### 5.6 内存跟踪机制

中间张量的数据指针由 `dl_graph_track` 接管：

```c
static void dl_graph_track(DLTensor* t) {
    if (dl_graph.no_grad) return;        // 推理模式不跟踪
    if (t->requires_grad) return;         // 参数不跟踪（参数自己管理生命周期）
    dl_graph.tracked_data[n_tracked++] = t->data;  // 保存数据指针
    t->owns_data = false;                 // 张量不再负责释放数据
}
```

`dl_graph_clear()` 释放所有跟踪的数据：
```c
void dl_graph_clear(void) {
    // 释放图节点
    for (int i = 0; i < n_nodes; i++) {
        node->output->graph_node = NULL;  // 断开张量与节点的关联
        free(node->extra);
        free(node);
    }
    // 释放跟踪的数据
    for (int i = 0; i < n_tracked; i++) {
        free(tracked_data[i]);
    }
}
```

---

## 6. dl_nn — 神经网络层

### 6.1 Linear 层

```c
typedef struct {
    DLTensor* weight;   // (out_features, in_features) — 注意不是 (in, out)
    DLTensor* bias;     // (out_features,) 或 NULL
} DLLinear;
```

**初始化：** 使用 **Kaiming 初始化** `std = sqrt(2/fan_in)`，这对 ReLU/GELU 激活来说收敛更快。

**前向传播流程：**
```
input: (batch, ..., in_features)
  ↓ reshape
flat: (total_batch, in_features)
  ↓ W^T = transpose(weight, 0, 1)
  ↓ matmul(flat, W^T)
out: (total_batch, out_features)
  ↓ + bias (广播)
  ↓ reshape
output: (batch, ..., out_features)
```

关键实现细节：在梯度模式下，`wt_c`（权重转置的连续拷贝）通过 `dl_graph_track_tensor` 跟踪，确保反向传播时仍然可用。

### 6.2 Embedding 层

```c
typedef struct {
    DLTensor* weight;   // (vocab_size, embed_dim) — 查找表
} DLEmbedding;
```

前向就是简单的内存拷贝：
```c
for (int i = 0; i < n; i++) {
    memcpy(out + i * embed_dim,
           weight + indices[i] * embed_dim,
           embed_dim * sizeof(float));
}
```

反向则是 **散射加法**（scatter add）：
```c
for (int i = 0; i < n; i++) {
    int idx = indices[i];
    for (int j = 0; j < embed_dim; j++)
        grad_weight[idx * embed_dim + j] += grad_out[i * embed_dim + j];
}
```

### 6.3 LayerNorm 层

初始化 gamma=1, beta=0。前向调用 `dl_ag_layer_norm`，反向在 autograd 中实现。

### 6.4 参数列表

`DLParamList` 是一个动态数组，支持自动扩容：
```c
void dl_paramlist_add(DLParamList* list, DLTensor* param) {
    if (list->n_params >= list->capacity) {
        list->capacity *= 2;
        list->params = realloc(list->params, ...);
    }
    list->params[list->n_params++] = param;
}
```

---

## 7. dl_transformer — Transformer 架构

### 7.1 多头注意力 (Multi-Head Attention)

```c
typedef struct {
    DLLinear* q_proj;         // Q 投影: d_model -> d_model
    DLLinear* k_proj;         // K 投影: d_model -> d_model
    DLLinear* v_proj;         // V 投影: d_model -> d_model
    DLLinear* o_proj;         // 输出投影: d_model -> d_model
    DLDropout* attn_dropout;
    DLDropout* resid_dropout;
    int n_heads, d_model, head_dim;
} DLMultiHeadAttention;
```

**前向流程：**

```
输入 x: (batch, seq, d_model)

1. Q = q_proj(x)  →  (batch, seq, d_model)
2. K = k_proj(x)  →  (batch, seq, d_model)
3. V = v_proj(x)  →  (batch, seq, d_model)

4. 重塑为多头: (batch, seq, n_heads, head_dim)
5. 转置: (batch, n_heads, seq, head_dim)

6. 注意力分数: scores = Q @ K^T / √head_dim
   (batch, n_heads, seq, seq)

7. 因果掩码 (可选):
   for i < seq:
     for j > i: scores[..., i, j] = -1e9   // 屏蔽未来位置

8. 注意力权重: weights = softmax(scores, dim=-1)
9. Dropout(weights)

10. 加权求和: output = weights @ V
    (batch, n_heads, seq, head_dim)

11. 转置回: (batch, seq, n_heads, head_dim)
12. 合并头: (batch, seq, d_model)
13. 输出投影: o_proj(output)
14. 残差 Dropout
```

**因果掩码** 确保位置 i 只能看到位置 0..i 的信息（自回归）：
```c
if (causal_mask) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len; j++) {
            scaled->data[...] = -1e9f;  // softmax 后趋近于 0
        }
    }
}
```

### 7.2 Transformer 块 (Pre-norm)

```
输入 x
  ├──→ LayerNorm ──→ MultiHeadAttention ──→ (+) ──→ x'
  └────────────────────────────────────────→ (+)

x'
  ├──→ LayerNorm ──→ Linear(d_ff) ──→ GELU ──→ Linear(d_model) ──→ Dropout ──→ (+) ──→ 输出
  └──────────────────────────────────────────────────────────────────────────→ (+)
```

**为什么用 Pre-norm？** 将 LayerNorm 放在子层之前（而非之后）可以稳定深层网络的训练。GPT-2 及后续大模型普遍采用。

### 7.3 完整模型

```
输入 tokens: [t0, t1, ..., tn]

1. Token Embedding:    token_embed(tokens)    →  (batch, seq, d_model)
2. Position Embedding: pos_embed([0,1,...,n])  →  (1, seq, d_model)
3. 相加:               tok_emb + pos_emb       →  (batch, seq, d_model)
4. Embedding Dropout

5. N 个 Transformer Block (循环)

6. Final LayerNorm
7. LM Head (Linear):   (batch, seq, d_model) → (batch, seq, vocab_size)

输出: logits (未归一化的对数概率)
```

### 7.4 损失计算

```c
DLTensor* dl_transformer_loss(model, tokens, batch_size, seq_len) {
    // 1. 前向传播获得 logits: (batch, seq, vocab)
    DLTensor* logits = dl_transformer_forward(model, tokens, ...);

    // 2. 展平为 (batch*seq, vocab)
    DLTensor* flat = dl_ag_reshape(logits, (batch*seq, vocab));

    // 3. 构造 targets: token[t+1] 是 token[t] 的目标
    targets[b*seq + t] = tokens[b*seq + t + 1];

    // 4. 计算交叉熵
    DLTensor* loss = dl_ag_cross_entropy(flat, targets, batch*seq, vocab);
}
```

### 7.5 文本生成

```c
int dl_transformer_generate_next(model, tokens, seq_len, temperature, top_k) {
    // 1. 关闭梯度 + 推理模式
    dl_set_no_grad(true);
    model->training = false;

    // 2. 前向得到最后位置的 logits
    DLTensor* logits = dl_transformer_forward(model, tokens, 1, seq_len);
    float* last = logits->data + (seq_len - 1) * vocab_size;

    // 3. 温度缩放: logits /= temperature
    //    温度越低，分布越尖锐（越确定性）

    // 4. Top-K 过滤: 只保留概率最高的 K 个 token
    //    其余设为 -1e9 (softmax后接近0)

    // 5. Softmax 得到概率分布

    // 6. 从分布中采样
    float r = random();
    float cum = 0;
    for (int i = 0; i < vocab; i++) {
        cum += probs[i];
        if (r < cum) return i;
    }
}
```

---

## 8. dl_optimizer — 优化器

### 8.1 学习率调度

**Warmup + 余弦退火：**

```
         warmup        cosine decay
lr  ─────/──────\─────────────────────
   0    w_steps                 total_steps
```

```c
float dl_scheduler_get_lr(DLLRScheduler* sched) {
    if (step < warmup_steps) {
        return base_lr * (step + 1) / warmup_steps;     // 线性增长
    } else {
        float progress = (step - warmup) / (total - warmup);
        return base_lr * 0.5 * (1 + cos(π * progress));  // 余弦衰减到 0
    }
}
```

### 8.2 AdamW 优化器

```c
void dl_adam_step(DLAdam* opt) {
    opt->t++;
    float lr = scheduler ? scheduler_lr : opt->lr;
    float bc1 = 1 - β1^t;   // 偏差校正系数
    float bc2 = 1 - β2^t;

    for each parameter p with gradient g:
        // 更新一阶矩（梯度的指数移动平均）
        m = β1 * m + (1 - β1) * g

        // 更新二阶矩（梯度平方的指数移动平均）
        v = β2 * v + (1 - β2) * g²

        // 偏差校正
        m̂ = m / bc1
        v̂ = v / bc2

        // 参数更新
        update = lr * m̂ / (√v̂ + ε)

        // AdamW: 解耦权重衰减
        if (adamw) update += lr * weight_decay * p

        p -= update
}
```

**Adam vs AdamW：** 标准 Adam 将 L2 正则化加到梯度上（`g += wd * p`），导致正则化强度与自适应学习率耦合。AdamW 直接从参数上减去（`p -= lr * wd * p`），效果更好。

### 8.3 梯度裁剪

```c
float dl_grad_clip_norm(DLParamList* params, float max_norm) {
    // 1. 计算全局梯度范数
    float total_norm = sqrt(Σ ||grad_i||²);

    // 2. 如果超过阈值，等比缩放
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6);
        for each grad: grad *= scale;
    }
    return total_norm;
}
```

---

## 9. dl_tokenizer — 分词器

### 9.1 字符级分词

```c
DLTokenizer* dl_tokenizer_create_char(const char* text) {
    // 预留特殊 token
    add("<pad>");  // ID=0  填充
    add("<unk>");  // ID=1  未知
    add("<bos>");  // ID=2  句首
    add("<eos>");  // ID=3  句尾

    // 扫描文本，每个唯一字符创建一个 token
    for (char c in text) {
        if (!seen[c]) {
            add(string(c));  // 如 "a" -> ID=4, "b" -> ID=5, ...
        }
    }
}
```

### 9.2 BPE 训练

**Byte Pair Encoding** 迭代合并最频繁的相邻 token 对：

```
初始: ['H', 'e', 'l', 'l', 'o', ' ', 'H', 'e', 'l', 'l', 'o']

统计最频繁对: ('l', 'l') 出现 2 次
合并: ['H', 'e', 'll', 'o', ' ', 'H', 'e', 'll', 'o']
新 token: "ll" -> vocab 中新增

统计: ('H', 'e') 出现 2 次
合并: ['He', 'll', 'o', ' ', 'He', 'll', 'o']
...

每次合并压缩序列长度，提高编码效率。
```

### 9.3 编码流程

```c
int* dl_tokenizer_encode(DLTokenizer* tok, const char* text, int* out_len) {
    // 1. 字符级编码
    for each char: ids[i] = find_char_token(char);

    // 2. 按顺序应用 BPE 合并规则
    for (int m = 0; m < n_merges; m++) {
        for (int i = 0; i < ids_len - 1; i++) {
            if (ids[i] == merge_a[m] && ids[i+1] == merge_b[m]) {
                ids[i] = merged_token_id;
                // 删除 ids[i+1]
            }
        }
    }
    return ids;
}
```

---

## 10. dl_dataloader — 数据加载器

### 10.1 数据组织

```
文本 token 序列: [t0, t1, t2, t3, t4, t5, t6, t7, t8, ...]

seq_len = 4, batch_size = 2:

Batch 0: [t0, t1, t2, t3] [t4, t5, t6, t7]    // 两个样本
Batch 1: [t4, t5, t6, t7] [t8, t9, t10, t11]   // 下两个样本
...

每个样本中: 输入 = [t0..t3], 目标 = [t1..t4]（next-token prediction）
```

### 10.2 Shuffle

Fisher-Yates 洗牌算法，打乱的是批次索引而非数据本身：
```c
for (int i = n - 1; i > 0; i--) {
    int j = random() * (i + 1);
    swap(shuffle_indices[i], shuffle_indices[j]);
}
```

---

## 11. dl_serialize — 模型序列化

### 11.1 自定义二进制格式

```
文件布局:
┌──────────────────────────────┐
│ Magic: 0x444C4D4C ("DLML")  │  4 bytes
│ Version: 1                   │  4 bytes
│ N_Params                     │  4 bytes
│ DLTransformerConfig          │  sizeof(config) bytes
├──────────────────────────────┤
│ TensorEntry[0]               │  name(128) + ndim(4) + shape + offset(8) + size(8)
│ TensorEntry[1]               │
│ ...                          │
├──────────────────────────────┤
│ float data[param_0]          │  连续的 float 数组
│ float data[param_1]          │
│ ...                          │
└──────────────────────────────┘
```

### 11.2 检查点格式

检查点在模型权重之外，还保存：
- 当前训练步数
- Adam 优化器的一阶矩 (m) 和二阶矩 (v)
- 优化器时间步 t

### 11.3 GGUF 读取器

GGUF 文件格式（llama.cpp 使用）：

```
GGUF 文件布局:
┌─────────────────────────┐
│ Magic: 0x46475547 "GGUF"│
│ Version                  │
│ N_Tensors               │
│ N_Metadata              │
├─────────────────────────┤
│ Metadata KV pairs       │  (跳过)
├─────────────────────────┤
│ Tensor Info entries:    │
│   - name (string)       │
│   - ndim, shape[]       │
│   - type (F32/F16/Q4..) │
│   - data offset         │
├───── 32-byte aligned ───┤
│ Tensor data             │
└─────────────────────────┘
```

**F16 到 F32 转换：**
```c
static float dl_f16_to_f32(uint16_t h) {
    // IEEE 754 半精度: 1位符号 + 5位指数 + 10位尾数
    // 转换为单精度: 1位符号 + 8位指数 + 23位尾数
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = ((h >> 10) & 0x1F);
    uint32_t mantissa = h & 0x3FF;
    // 指数偏移转换: exp_f32 = exp_f16 + (127 - 15)
    // 尾数左移: mantissa_f32 = mantissa_f16 << 13
}
```

---

## 12. 内存管理机制详解

### 12.1 三层内存管理

```
层级 1: 参数内存（长期）
  - 由 DLLinear, DLEmbedding 等层创建
  - 生命周期 = 模型存在期间
  - owns_data = true, requires_grad = true
  - 梯度张量也是长期存在

层级 2: 中间张量数据（每步释放）
  - 由 dl_ag_* 操作创建
  - 数据指针由 dl_graph.tracked_data 管理
  - owns_data = false（被 graph_track 接管）
  - dl_graph_clear() 批量释放

层级 3: 张量结构体（即时释放）
  - 小的 DLTensor 结构体（约 100 bytes）
  - 通过 dl_tensor_free 引用计数管理
  - 与数据独立——结构体可以先释放，数据在 graph_clear 时释放
```

### 12.2 训练循环中的内存流

```
每一步:
  dl_paramlist_zero_grad()     // 清零参数梯度（不分配新内存）
  dl_graph_clear()             // 释放上一步的中间张量数据 + 图节点

  forward()                    // 分配新的中间张量（数据 tracked）
  dl_backward()                // 分配梯度张量（中间张量的梯度也 tracked）

  optimizer_step()             // 原地更新参数

// 下一步的 graph_clear() 释放本步的所有中间数据
```

### 12.3 防止双重释放

```c
void dl_graph_track(DLTensor* t) {
    tracked_data[n++] = t->data;   // 保存数据指针
    t->owns_data = false;           // 张量不再负责释放数据
}

// dl_tensor_free(t) 时:
//   owns_data == false → 不释放 data
//   只释放 DLTensor 结构体

// dl_graph_clear() 时:
//   遍历 tracked_data，释放所有数据指针
```

---

## 13. 计算图与反向传播详解

### 13.1 前向时构建图

一个简单的 `y = relu(x @ W + b)` 的计算图：

```
     x (input)     W (param)
         \           /
      dl_ag_matmul (node 0)
              |
          matmul_out
              |          b (param)
      dl_ag_add (node 1)    /
              |            /
          add_out --------
              |
      dl_ag_relu (node 2)
              |
          relu_out = y
```

### 13.2 反向时遍历图

```
步骤 1: 拓扑排序 (DFS后序)
  sorted = [node_0(matmul), node_1(add), node_2(relu)]

步骤 2: 逆序遍历
  - backward_relu:  dL/d(add_out) = dL/dy * (add_out > 0 ? 1 : 0)
  - backward_add:   dL/d(matmul_out) += dL/d(add_out)
                     dL/d(b) += sum(dL/d(add_out))
  - backward_matmul: dL/d(x) = dL/d(matmul_out) @ W^T
                      dL/d(W) = x^T @ dL/d(matmul_out)
```

### 13.3 梯度累加

参数可能在多个运算中使用（如权重共享或多次前向）。梯度使用 `+=`（累加）而非 `=`（覆盖）：
```c
DLTensor* ga = dl_tensor_ensure_grad(a);  // 分配或获取已有梯度
dl_accumulate_grad(ga, computed_grad);      // 累加
```

---

## 14. SIMD 加速实现

### 14.1 三级降级策略

```c
// 级别 1: AVX2 (256位, 8 x float32)
#if DL_USE_AVX2
    __m256 va = _mm256_loadu_ps(a + i);    // 加载8个float
    __m256 vb = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));

// 级别 2: SSE2 (128位, 4 x float32)
#elif DL_USE_SSE2
    __m128 va = _mm_loadu_ps(a + i);       // 加载4个float
    __m128 vb = _mm_loadu_ps(b + i);
    _mm_storeu_ps(out + i, _mm_add_ps(va, vb));

// 级别 3: 标量 (1 x float32)
#endif
    out[i] = a[i] + b[i];                  // 处理剩余元素
```

### 14.2 矩阵乘法中的 SIMD

在分块矩阵乘法的最内层循环中使用 SIMD：
```c
float a_ik = A[i * K + k];
__m256 va = _mm256_set1_ps(a_ik);          // 广播标量到8路
for (j = j0; j + 7 < jmax; j += 8) {
    __m256 vb = _mm256_loadu_ps(&B[k*N+j]); // 加载 B 的一行
    __m256 vc = _mm256_loadu_ps(&C[i*N+j]); // 加载 C 的当前值
    vc = _mm256_add_ps(vc, _mm256_mul_ps(va, vb));  // FMA: C += a * B
    _mm256_storeu_ps(&C[i*N+j], vc);
}
```

这是经典的 **广播-乘-加** 模式，将 A 的一个元素广播后与 B 的一行相乘，累加到 C。

### 14.3 点积中的 SIMD

```c
float dl_vec_dot(const float* a, const float* b, int n) {
    __m256 vsum = _mm256_setzero_ps();     // 累加器
    for (i = 0; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));  // 8路并行乘加
    }
    // 水平归约: 将8个部分和相加
    float buf[8];
    _mm256_storeu_ps(buf, vsum);
    float sum = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    // 处理剩余元素
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
```

### 14.4 性能数据

在 AVX2 CPU 上的参考性能（4层, 128维, batch=4, seq=64）：
- 训练速度: ~1400 tokens/sec
- 单步时间: ~180ms
- 矩阵乘法占总时间的 60%+
- SIMD 加速比（vs 纯标量）: 约 3-5x
