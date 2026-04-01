# DeepLearn-C 使用教程

## 目录

1. [项目简介](#1-项目简介)
2. [环境要求与编译](#2-环境要求与编译)
3. [快速开始：训练你的第一个模型](#3-快速开始训练你的第一个模型)
4. [核心 API 使用指南](#4-核心-api-使用指南)
   - 4.1 [张量操作](#41-张量操作)
   - 4.2 [数学运算](#42-数学运算)
   - 4.3 [自动微分](#43-自动微分)
   - 4.4 [神经网络层](#44-神经网络层)
   - 4.5 [Transformer 模型](#45-transformer-模型)
   - 4.6 [优化器](#46-优化器)
   - 4.7 [分词器与数据加载](#47-分词器与数据加载)
   - 4.8 [模型序列化](#48-模型序列化)
5. [训练完整流程详解](#5-训练完整流程详解)
6. [推理与文本生成](#6-推理与文本生成)
7. [自定义模型配置](#7-自定义模型配置)
8. [性能调优指南](#8-性能调优指南)
9. [GGUF 权重导入](#9-gguf-权重导入)
10. [常见问题](#10-常见问题)

---

## 1. 项目简介

DeepLearn-C 是一个用纯 C 语言编写的深度学习框架，专注于 Transformer 架构的预训练和推理。无任何外部依赖，仅使用 C99 标准库和可选的 SIMD 指令集加速。

**核心特性：**
- 完整的 N 维张量运算库，支持广播语义
- 反向模式自动微分（Autograd），支持 15 种可微分运算
- GPT-2 风格的 Transformer 解码器，支持多头注意力和因果掩码
- Adam/AdamW 优化器 + 余弦退火学习率调度
- 字符级 + BPE 分词器
- 自定义二进制格式 + GGUF 格式权重读取
- AVX2/SSE2 SIMD 自动检测加速

---

## 2. 环境要求与编译

### 2.1 环境要求

| 项目 | 要求 |
|------|------|
| 编译器 | GCC 4.9+ 或 Clang 3.5+（需支持 C99） |
| 操作系统 | Linux / macOS / WSL |
| 可选 | 支持 AVX2 的 CPU（2013 年后的 Intel/AMD 处理器） |

### 2.2 使用 Make 编译

```bash
cd /root/test/deeplearn
make            # 编译全部目标
make test       # 编译并运行测试
make clean      # 清理构建产物
```

### 2.3 使用 CMake 编译

```bash
cd /root/test/deeplearn
cmake -B build
cmake --build build
./build/test_tensor     # 运行测试
```

### 2.4 编译产物

```
build/
├── libdeeplearn.a      # 静态库
├── test_tensor         # 单元测试
├── train_gpt2          # 预训练示例
└── inference            # 推理示例
```

### 2.5 SIMD 加速确认

编译输出中会显示 SIMD 支持信息：
```
# Make 方式会自动添加 -mavx2 -mfma 或 -msse2
# CMake 方式会输出 "AVX2 support enabled" 或 "SSE2 support enabled"
```

---

## 3. 快速开始：训练你的第一个模型

### 3.1 使用内置文本训练

```bash
# 最简单的方式，使用内置样本文本
./build/train_gpt2

# 指定训练步数和保存路径
./build/train_gpt2 --steps 200 --save model.bin
```

### 3.2 使用自定义文本训练

```bash
# 准备文本文件
echo "你的训练文本..." > data.txt

# 使用自定义文本训练
./build/train_gpt2 data.txt --steps 500 --save model.bin
```

### 3.3 全部训练参数

```
./build/train_gpt2 [text_file] [options]

选项：
  --steps N       训练步数          (默认: 200)
  --lr F          学习率            (默认: 3e-4)
  --batch N       批大小            (默认: 4)
  --seq N         序列长度          (默认: 64)
  --layers N      Transformer 层数  (默认: 4)
  --heads N       注意力头数        (默认: 4)
  --dim N         模型维度          (默认: 128)
  --save PATH     模型保存路径
```

### 3.4 预期输出

```
=== DeepLearn-C GPT-2 Pre-training ===
Config: layers=4, heads=4, d_model=128, d_ff=512
Training: steps=200, lr=3.0e-04, batch=4, seq_len=64
...
--- Training ---
step    1/200 | loss: 5.9119 | lr: 3.00e-05 | grad_norm: 10.65 | 225ms/step | 1140 tok/s
step   50/200 | loss: 5.0032 | lr: 2.80e-04 | grad_norm:  8.92 | 177ms/step | 1442 tok/s
step  200/200 | loss: 4.1253 | lr: 0.00e+00 | grad_norm:  9.11 | 185ms/step | 1385 tok/s

Training complete! Average time: 180.3 ms/step

--- Generation ---
Generated: The qowther worqe is f a ...
```

---

## 4. 核心 API 使用指南

### 4.1 张量操作

张量是框架的基础数据结构，支持 N 维浮点数组（最多 8 维）。

#### 创建张量

```c
#include "dl_tensor.h"

// 初始化随机数生成器（全局，调用一次即可）
dl_rng_init(42);

// 创建指定形状的零张量
int shape[] = {3, 4};
DLTensor* zeros = dl_tensor_zeros(shape, 2);       // (3, 4) 全零

// 创建全一张量
DLTensor* ones = dl_tensor_ones(shape, 2);          // (3, 4) 全一

// 创建随机张量（均匀分布 [0,1)）
DLTensor* rnd = dl_tensor_rand(shape, 2);

// 创建正态分布随机张量
DLTensor* normal = dl_tensor_randn(shape, 2, 0.0f, 1.0f);  // 均值0, 标准差1

// 从已有数据创建张量
float data[] = {1, 2, 3, 4, 5, 6};
int s[] = {2, 3};
DLTensor* t = dl_tensor_from_data(data, s, 2);     // (2, 3) 从数据创建

// 创建标量张量
DLTensor* scalar = dl_tensor_scalar(3.14f);          // 0维标量

// 克隆（深拷贝）
DLTensor* copy = dl_tensor_clone(t);
```

#### 元素级运算

```c
DLTensor* a = dl_tensor_rand((int[]){2, 3}, 2);
DLTensor* b = dl_tensor_rand((int[]){2, 3}, 2);

// 四则运算（返回新张量）
DLTensor* c = dl_tensor_add(a, b);    // c = a + b
DLTensor* d = dl_tensor_sub(a, b);    // d = a - b
DLTensor* e = dl_tensor_mul(a, b);    // e = a * b（逐元素）
DLTensor* f = dl_tensor_div(a, b);    // f = a / b

// 标量运算
DLTensor* g = dl_tensor_scale(a, 2.0f);       // g = a * 2
DLTensor* h = dl_tensor_add_scalar(a, 1.0f);  // h = a + 1

// 原地运算（修改 a 本身）
dl_tensor_add_(a, b);      // a += b
dl_tensor_scale_(a, 0.5f); // a *= 0.5
dl_tensor_fill_(a, 0.0f);  // a[:] = 0
dl_tensor_zero_(a);        // a[:] = 0（等效于 fill_(0)）
```

#### 形状操作

```c
DLTensor* t = dl_tensor_rand((int[]){2, 3, 4}, 3);  // (2, 3, 4)

// 重塑（返回视图，不拷贝数据）
DLTensor* r = dl_tensor_reshape(t, (int[]){6, 4}, 2);  // (6, 4)

// 转置（交换两个维度）
DLTensor* tr = dl_tensor_transpose(t, 0, 1);  // (3, 2, 4)

// 检查是否连续存储
bool contig = dl_tensor_is_contiguous(tr);  // false（转置后通常不连续）

// 强制连续化（如果已连续则返回引用，否则深拷贝）
DLTensor* c = dl_tensor_contiguous(tr);
```

#### 规约运算

```c
DLTensor* t = dl_tensor_rand((int[]){3, 4}, 2);

// 沿指定维度求和
DLTensor* s = dl_tensor_sum(t, 1, false);   // (3,) — 沿 dim=1 求和，不保留维度
DLTensor* sk = dl_tensor_sum(t, 1, true);   // (3, 1) — 保留维度

// 沿维度求均值
DLTensor* m = dl_tensor_mean(t, 0, false);  // (4,)

// 全局规约
float total = dl_tensor_sum_all(t);          // 所有元素之和
float max_v = dl_tensor_max_all(t);          // 全局最大值
```

#### 广播机制

```c
// 不同形状的张量可以自动广播
DLTensor* a = dl_tensor_rand((int[]){2, 3}, 2);  // (2, 3)
DLTensor* b = dl_tensor_rand((int[]){1, 3}, 2);  // (1, 3) — 自动广播到 (2, 3)
DLTensor* c = dl_tensor_add(a, b);                // (2, 3) — 正确广播
```

#### 内存管理

```c
// 释放张量（引用计数机制，计数归零时真正释放）
dl_tensor_free(t);

// 增加引用计数（用于视图共享）
DLTensor* ref = dl_tensor_ref(t);  // t->ref_count++

// 打印张量信息
dl_tensor_print(t, "my_tensor");
// 输出: my_tensor: shape=[2,3], data=[0.1234, 0.5678, ...]
```

### 4.2 数学运算

```c
#include "dl_ops.h"

// 矩阵乘法（自动分块 + SIMD 加速）
DLTensor* a = dl_tensor_rand((int[]){2, 3}, 2);  // (2, 3)
DLTensor* b = dl_tensor_rand((int[]){3, 4}, 2);  // (3, 4)
DLTensor* c = dl_matmul(a, b);                    // (2, 4)

// 批量矩阵乘法
DLTensor* ba = dl_tensor_rand((int[]){8, 32, 64}, 3);  // (8, 32, 64)
DLTensor* bb = dl_tensor_rand((int[]){8, 64, 16}, 3);  // (8, 64, 16)
DLTensor* bc = dl_bmm(ba, bb);                          // (8, 32, 16)

// Softmax（沿最后一维，数值稳定实现）
DLTensor* logits = dl_tensor_rand((int[]){2, 10}, 2);
DLTensor* probs = dl_softmax(logits, -1);    // 每行概率和为 1

// Log-Softmax
DLTensor* log_p = dl_log_softmax(logits, -1);

// 层归一化
DLTensor* x = dl_tensor_rand((int[]){2, 64}, 2);
DLTensor* gamma = dl_tensor_ones((int[]){64}, 1);   // 可学习缩放
DLTensor* beta = dl_tensor_zeros((int[]){64}, 1);   // 可学习偏移
DLTensor* normed = dl_layer_norm(x, gamma, beta, 1e-5f);

// 激活函数
DLTensor* g = dl_gelu(x);    // GELU (GPT-2 使用)
DLTensor* r = dl_relu(x);    // ReLU
DLTensor* s = dl_silu(x);    // SiLU / Swish

// 交叉熵损失
DLTensor* logits2 = dl_tensor_rand((int[]){4, 100}, 2);  // (batch=4, vocab=100)
int targets[] = {5, 23, 42, 7};                            // 目标 token id
DLTensor* loss = dl_cross_entropy_loss(logits2, targets, 4, 100);

// Embedding 查找
DLTensor* emb_weight = dl_tensor_rand((int[]){1000, 64}, 2);  // (vocab, dim)
int indices[] = {5, 10, 15};
DLTensor* emb_out = dl_embedding_forward(emb_weight, indices, 3);  // (3, 64)
```

### 4.3 自动微分

自动微分（Autograd）是训练的核心——它自动计算损失函数对每个参数的梯度。

```c
#include "dl_autograd.h"

// 初始化计算图
dl_graph_init();

// 创建需要梯度的张量（参数）
DLTensor* w = dl_tensor_randn((int[]){3, 4}, 2, 0.0f, 0.1f);
dl_tensor_set_requires_grad(w, true);

// 使用 dl_ag_* 系列函数进行前向计算（自动记录计算图）
DLTensor* x = dl_tensor_rand((int[]){2, 3}, 2);
DLTensor* y = dl_ag_matmul(x, w);          // y = x @ w  (记录到图中)
DLTensor* z = dl_ag_gelu(y);               // z = gelu(y) (记录到图中)
DLTensor* loss = dl_ag_scale(z, 1.0f);     // 简单的 loss

// 反向传播
dl_backward(loss);

// 访问梯度
printf("w.grad[0] = %f\n", w->grad->data[0]);

// 清理计算图（释放中间张量，为下一次迭代做准备）
dl_graph_clear();
```

#### 可微分操作一览

```c
DLTensor* dl_ag_matmul(a, b);                        // 矩阵乘法
DLTensor* dl_ag_add(a, b);                           // 加法
DLTensor* dl_ag_mul(a, b);                           // 逐元素乘法
DLTensor* dl_ag_scale(a, scalar);                    // 标量乘法
DLTensor* dl_ag_gelu(x);                             // GELU 激活
DLTensor* dl_ag_relu(x);                             // ReLU 激活
DLTensor* dl_ag_silu(x);                             // SiLU 激活
DLTensor* dl_ag_softmax(x, dim);                     // Softmax
DLTensor* dl_ag_layer_norm(x, gamma, beta, eps);     // 层归一化
DLTensor* dl_ag_embedding(weight, indices, n);       // Embedding 查找
DLTensor* dl_ag_cross_entropy(logits, targets, B, V);// 交叉熵损失
DLTensor* dl_ag_dropout(x, p, training);             // Dropout
DLTensor* dl_ag_transpose(x, dim0, dim1);            // 转置
DLTensor* dl_ag_reshape(x, shape, ndim);             // 重塑
```

#### 禁用梯度（推理模式）

```c
dl_set_no_grad(true);   // 禁用梯度跟踪，节省内存
// ... 执行推理 ...
dl_set_no_grad(false);  // 恢复梯度跟踪
```

### 4.4 神经网络层

```c
#include "dl_nn.h"

// === Linear 全连接层 ===
DLLinear* fc = dl_linear_create(128, 64, true);  // in=128, out=64, 有bias
DLTensor* x = dl_tensor_rand((int[]){4, 128}, 2);
DLTensor* y = dl_linear_forward(fc, x);          // (4, 64)
// 支持 3D+ 输入：(batch, seq, features) -> (batch, seq, out_features)

// === Embedding 嵌入层 ===
DLEmbedding* emb = dl_embedding_create(10000, 256);  // vocab=10000, dim=256
int tokens[] = {42, 100, 7};
DLTensor* e = dl_embedding_lookup(emb, tokens, 3);   // (3, 256)

// === LayerNorm 层归一化 ===
DLLayerNorm* ln = dl_layernorm_create(256, 1e-5f);   // dim=256
DLTensor* normed = dl_layernorm_forward(ln, x);

// === Dropout ===
DLDropout* drop = dl_dropout_create(0.1f);  // 10% 丢弃率
drop->training = true;                       // 训练模式
DLTensor* dropped = dl_dropout_forward(drop, x);
drop->training = false;                      // 推理模式（不丢弃）

// === 参数收集 ===
DLParamList* params = dl_paramlist_create(64);
dl_paramlist_add_linear(params, fc);
dl_paramlist_add_embedding(params, emb);
dl_paramlist_add_layernorm(params, ln);

int total = dl_paramlist_total_params(params);  // 参数总数
dl_paramlist_zero_grad(params);                 // 清零所有梯度

// === 释放 ===
dl_linear_free(fc);
dl_embedding_free(emb);
dl_layernorm_free(ln);
dl_dropout_free(drop);
dl_paramlist_free(params);
```

### 4.5 Transformer 模型

```c
#include "dl_transformer.h"

// 配置模型
DLTransformerConfig config = {
    .vocab_size     = 10000,  // 词表大小
    .max_seq_len    = 512,    // 最大序列长度
    .n_layers       = 6,      // Transformer 层数
    .n_heads        = 6,      // 注意力头数
    .d_model        = 384,    // 模型维度
    .d_ff           = 1536,   // FFN 中间层大小（通常 4 * d_model）
    .dropout_p      = 0.1f,   // Dropout 概率
    .layer_norm_eps = 1e-5f   // LayerNorm epsilon
};

// 创建模型
DLTransformerModel* model = dl_transformer_create(config);
printf("参数量: %d\n", dl_paramlist_total_params(model->params));

// 前向传播（获取 logits）
int tokens[] = {1, 42, 100, 7, 256};
DLTensor* logits = dl_transformer_forward(model, tokens, 1, 5);  // (1, 5, vocab)

// 计算损失（内置 next-token prediction）
DLTensor* loss = dl_transformer_loss(model, tokens, 1, 5);
printf("Loss: %f\n", loss->data[0]);

// 设置训练/推理模式
dl_transformer_set_training(model, true);   // 训练模式（启用 dropout）
dl_transformer_set_training(model, false);  // 推理模式（关闭 dropout）

// 文本生成（采样下一个 token）
int next_token = dl_transformer_generate_next(
    model,
    tokens, 5,    // 输入 token 序列
    0.8f,         // temperature (越低越确定)
    40            // top-k (仅从概率最高的 k 个中采样)
);

// 释放
dl_transformer_free(model);
```

### 4.6 优化器

```c
#include "dl_optimizer.h"

// === AdamW 优化器（推荐用于 Transformer 训练）===
DLAdam* optimizer = dl_adam_create(
    model->params,   // 参数列表
    3e-4f,           // 学习率
    0.9f,            // beta1（一阶矩衰减）
    0.999f,          // beta2（二阶矩衰减）
    1e-8f,           // epsilon（数值稳定）
    0.01f,           // weight decay（权重衰减）
    true             // true = AdamW（解耦权重衰减）
);

// === 学习率调度器（Warmup + 余弦退火）===
DLLRScheduler* scheduler = dl_scheduler_create(
    3e-4f,    // 基础学习率
    100,      // warmup 步数
    10000     // 总训练步数
);
dl_adam_set_scheduler(optimizer, scheduler);

// === 训练循环 ===
for (int step = 0; step < 10000; step++) {
    dl_paramlist_zero_grad(model->params);  // 1. 清零梯度
    dl_graph_clear();                        // 2. 清理计算图

    DLTensor* loss = dl_transformer_loss(model, batch, batch_size, seq_len);
    dl_backward(loss);                       // 3. 反向传播

    float norm = dl_grad_clip_norm(model->params, 1.0f);  // 4. 梯度裁剪
    dl_adam_step(optimizer);                  // 5. 参数更新
}

// === SGD 优化器（可选）===
DLSGD* sgd = dl_sgd_create(model->params, 0.01f, 0.9f, 1e-4f);
// lr=0.01, momentum=0.9, weight_decay=1e-4
dl_sgd_step(sgd);

// 释放
dl_adam_free(optimizer);  // scheduler 会一起释放
dl_sgd_free(sgd);
```

### 4.7 分词器与数据加载

```c
#include "dl_tokenizer.h"
#include "dl_dataloader.h"

// === 从文本创建字符级分词器 ===
const char* text = "Hello, world! This is a test.";
DLTokenizer* tok = dl_tokenizer_create_char(text);
// 自动创建: <pad>(0), <unk>(1), <bos>(2), <eos>(3), 'H'(4), 'e'(5)...

// === 训练 BPE 合并 ===
dl_tokenizer_train_bpe(tok, text, 50);  // 学习 50 个合并规则
printf("词表大小: %d\n", tok->vocab_size);

// === 编码文本 ===
int n_tokens;
int* token_ids = dl_tokenizer_encode(tok, "Hello", &n_tokens);
// token_ids = [4, 5, 6, 6, 7] 或 BPE 合并后的更短序列

// === 解码回文本 ===
char* decoded = dl_tokenizer_decode(tok, token_ids, n_tokens);
printf("Decoded: %s\n", decoded);

// === 保存/加载词表 ===
dl_tokenizer_save(tok, "vocab.txt");
DLTokenizer* loaded = dl_tokenizer_load("vocab.txt");

// === 数据加载器 ===
// 从文本文件创建
DLDataLoader* loader = dl_dataloader_create("train.txt", tok, 4, 64);
// batch_size=4, seq_len=64

// 或从字符串创建
DLDataLoader* loader2 = dl_dataloader_from_text(text, tok, 2, 32);

printf("总 token 数: %d\n", loader->total_tokens);
printf("每 epoch 批次数: %d\n", loader->n_batches);

// === 迭代数据 ===
for (int epoch = 0; epoch < 10; epoch++) {
    dl_dataloader_shuffle(loader);    // 打乱批次顺序
    dl_dataloader_reset(loader);      // 重置到开头

    while (!dl_dataloader_epoch_done(loader)) {
        const int* batch = dl_dataloader_next_batch(loader);
        if (!batch) break;
        // batch 包含 batch_size * seq_len 个连续 token
        // 用于 next-token prediction 训练
    }
}

// 释放
free(token_ids);
free(decoded);
dl_tokenizer_free(tok);
dl_dataloader_free(loader);
```

### 4.8 模型序列化

```c
#include "dl_serialize.h"

// === 保存模型权重（自定义二进制格式）===
dl_save_model(model, "model.bin");

// === 加载模型权重 ===
int ret = dl_load_model(model, "model.bin");
if (ret != 0) printf("加载失败: %d\n", ret);

// === 保存完整训练检查点（含优化器状态）===
dl_save_checkpoint(model, optimizer, step, "checkpoint.bin");

// === 恢复检查点 ===
int resume_step;
dl_load_checkpoint(model, optimizer, &resume_step, "checkpoint.bin");
printf("从第 %d 步恢复\n", resume_step);

// === 加载 GGUF 格式权重 ===
GGUFFile* gguf = dl_gguf_load("model.gguf");
if (gguf) {
    printf("GGUF 张量数: %llu\n", gguf->n_tensors);

    // 按名称查找张量
    GGUFTensor* wt = dl_gguf_find_tensor(gguf, "token_embd.weight");
    if (wt) printf("Found: %s, size=%d\n", wt->name, wt->size);

    // 尝试将权重加载到模型中
    int loaded = dl_gguf_load_into_model(gguf, model);
    printf("成功加载 %d 个张量\n", loaded);

    dl_gguf_free(gguf);
}
```

---

## 5. 训练完整流程详解

以下是从零开始预训练一个 GPT 模型的完整代码：

```c
#include "dl_serialize.h"
#include "dl_dataloader.h"

int main(void) {
    // ---- 第 1 步：初始化 ----
    dl_rng_init(42);        // 设置随机种子，确保可复现
    dl_graph_init();         // 初始化计算图

    // ---- 第 2 步：准备分词器 ----
    const char* text = "... 你的训练文本 ...";
    DLTokenizer* tok = dl_tokenizer_create_char(text);
    dl_tokenizer_train_bpe(tok, text, 100);  // 可选的 BPE 合并

    // ---- 第 3 步：创建数据加载器 ----
    int batch_size = 4, seq_len = 64;
    DLDataLoader* loader = dl_dataloader_from_text(text, tok, batch_size, seq_len);

    // ---- 第 4 步：配置并创建模型 ----
    DLTransformerConfig config = {
        .vocab_size     = tok->vocab_size,
        .max_seq_len    = seq_len + 1,
        .n_layers       = 4,
        .n_heads        = 4,
        .d_model        = 128,
        .d_ff           = 512,
        .dropout_p      = 0.1f,
        .layer_norm_eps = 1e-5f
    };
    DLTransformerModel* model = dl_transformer_create(config);

    // ---- 第 5 步：创建优化器 ----
    int total_steps = 1000;
    DLAdam* opt = dl_adam_create(model->params,
        3e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, true);
    DLLRScheduler* sched = dl_scheduler_create(3e-4f, total_steps/10, total_steps);
    dl_adam_set_scheduler(opt, sched);

    // ---- 第 6 步：训练循环 ----
    for (int step = 0; step < total_steps; step++) {
        // 获取数据批次
        if (dl_dataloader_epoch_done(loader)) {
            dl_dataloader_reset(loader);
            dl_dataloader_shuffle(loader);
        }
        const int* batch = dl_dataloader_next_batch(loader);
        if (!batch) { dl_dataloader_reset(loader); continue; }

        // 前向 + 反向
        dl_paramlist_zero_grad(model->params);
        dl_graph_clear();
        dl_transformer_set_training(model, true);

        DLTensor* loss = dl_transformer_loss(model, batch, batch_size, seq_len);
        dl_backward(loss);

        // 梯度裁剪 + 优化器更新
        dl_grad_clip_norm(model->params, 1.0f);
        dl_adam_step(opt);

        // 打印日志
        if ((step + 1) % 10 == 0) {
            printf("step %d | loss: %.4f\n", step + 1, loss->data[0]);
        }
    }

    // ---- 第 7 步：保存模型 ----
    dl_save_model(model, "trained_model.bin");

    // ---- 第 8 步：生成文本 ----
    dl_transformer_set_training(model, false);
    int prompt[] = {4, 5, 6};  // 起始 token
    int generated[103];
    memcpy(generated, prompt, 3 * sizeof(int));

    for (int i = 0; i < 100; i++) {
        dl_graph_clear();
        int next = dl_transformer_generate_next(model, generated, 3 + i, 0.8f, 40);
        generated[3 + i] = next;
    }
    char* output = dl_tokenizer_decode(tok, generated, 103);
    printf("Generated: %s\n", output);

    // ---- 清理 ----
    free(output);
    dl_graph_clear();
    dl_adam_free(opt);
    dl_transformer_free(model);
    dl_dataloader_free(loader);
    dl_tokenizer_free(tok);
    return 0;
}
```

**编译自定义程序：**
```bash
gcc -std=c99 -O2 -Iinclude -mavx2 -mfma my_train.c -Lbuild -ldeeplearn -lm -o my_train
```

---

## 6. 推理与文本生成

```bash
# 使用训练好的模型生成文本
./build/inference model.bin --prompt "Once upon" --length 200 --temp 0.7 --topk 50

# 参数说明
#   --prompt TEXT   起始文本
#   --length N      生成 token 数（默认 200）
#   --temp F        温度（默认 0.8，越低越确定）
#   --topk N        Top-K 采样（默认 40）
#   --vocab PATH    词表文件路径
```

**温度参数的影响：**

| 温度 | 效果 |
|------|------|
| 0.1 | 非常确定性，几乎贪心解码 |
| 0.5 | 较低随机性，较连贯 |
| 0.8 | 中等随机性（推荐） |
| 1.0 | 标准采样 |
| 1.5 | 高随机性，更富创意但可能不连贯 |

---

## 7. 自定义模型配置

### 推荐配置参考

| 规模 | layers | heads | d_model | d_ff | 参数量 |
|------|--------|-------|---------|------|--------|
| 极小 | 2 | 2 | 64 | 256 | ~50K |
| 小 | 4 | 4 | 128 | 512 | ~840K |
| 中 | 6 | 6 | 384 | 1536 | ~10M |
| 大 | 12 | 12 | 768 | 3072 | ~85M |

**注意事项：**
- `d_model` 必须能被 `n_heads` 整除
- `d_ff` 通常设为 `4 * d_model`
- `max_seq_len` 应 >= `seq_len + 1`
- 较大模型需要更多训练步数和数据

---

## 8. 性能调优指南

### 8.1 训练速度

- **批大小（batch）：** 增大批大小可以提高 GPU/CPU 利用率，但增加内存
- **序列长度（seq）：** 注意力计算是 O(seq^2)，减小序列长度可显著加速
- **SIMD：** 确保编译时启用了 AVX2（`-mavx2 -mfma`）
- **编译优化：** 使用 `-O2` 或 `-O3`

### 8.2 内存占用

- 每一步的中间张量由计算图跟踪，`dl_graph_clear()` 释放
- 参数内存 = `总参数数 * 4 bytes`（float32）
- Adam 优化器额外消耗 `2 * 参数内存`（一阶和二阶矩）
- 减少层数或维度可显著降低内存

### 8.3 训练稳定性

- **梯度裁剪：** `dl_grad_clip_norm(params, 1.0f)` 防止梯度爆炸
- **学习率预热：** 前 10% 步数线性增长学习率
- **权重衰减：** 使用 AdamW 而非 Adam，weight_decay=0.01
- **Dropout：** 训练时 0.1，推理时设为 0

---

## 9. GGUF 权重导入

GGUF 是 llama.cpp 使用的模型格式。本框架支持读取 F32 和 F16 精度的 GGUF 文件。

```c
// 加载 GGUF 文件
GGUFFile* gguf = dl_gguf_load("llama-7b.gguf");

// 查看所有张量
for (uint64_t i = 0; i < gguf->n_tensors; i++) {
    GGUFTensor* t = &gguf->tensors[i];
    printf("%s: type=%d, shape=[", t->name, t->type);
    for (int d = 0; d < t->ndim; d++)
        printf("%lld%s", t->shape[d], d < t->ndim - 1 ? "," : "");
    printf("]\n");
}

// 按名称查找特定张量
GGUFTensor* emb = dl_gguf_find_tensor(gguf, "token_embd.weight");

// 自动加载到模型（按参数大小匹配）
int loaded = dl_gguf_load_into_model(gguf, model);

dl_gguf_free(gguf);
```

**支持的数据类型：**
- `GGUF_TYPE_F32` (0) — 完整支持
- `GGUF_TYPE_F16` (1) — 自动转换为 F32
- 量化类型 (Q4_0, Q8_0 等) — 读取但填零（不支持反量化）

---

## 10. 常见问题

**Q: 训练时出现 "graph full" 错误？**
A: 模型太大或序列太长，超过了计算图节点上限。在 `dl_common.h` 中增大 `DL_MAX_GRAPH_NODES`。

**Q: 如何在自己的 C 项目中使用？**
A: 将 `include/` 和 `build/libdeeplearn.a` 复制到你的项目中，编译时添加 `-Iinclude -Lpath -ldeeplearn -lm`。

**Q: 训练 loss 不下降？**
A: 检查学习率是否合适（通常 1e-4 ~ 3e-4），确保 `dl_backward()` 被调用，确保梯度裁剪阈值不太小。

**Q: 能加载 Hugging Face 的模型吗？**
A: 不能直接加载。需要先用工具将模型转换为 GGUF 格式（F32 精度），然后用 `dl_gguf_load` 读取。

**Q: 支持 GPU 吗？**
A: 当前版本仅支持 CPU（含 SIMD 加速）。可通过修改底层张量运算对接 CUDA。
