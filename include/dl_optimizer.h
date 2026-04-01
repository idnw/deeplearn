#ifndef DL_OPTIMIZER_H
#define DL_OPTIMIZER_H

#include "dl_nn.h"

/* Learning rate scheduler */
typedef struct {
    float base_lr;
    int warmup_steps;
    int total_steps;
    int current_step;
} DLLRScheduler;

DLLRScheduler* dl_scheduler_create(float base_lr, int warmup_steps, int total_steps);
void dl_scheduler_free(DLLRScheduler* sched);
float dl_scheduler_get_lr(DLLRScheduler* sched);
void dl_scheduler_step(DLLRScheduler* sched);

/* SGD optimizer */
typedef struct {
    DLParamList* params;
    float lr;
    float momentum;
    float weight_decay;
    DLTensor** velocity; /* momentum buffers */
} DLSGD;

DLSGD* dl_sgd_create(DLParamList* params, float lr, float momentum, float weight_decay);
void dl_sgd_free(DLSGD* opt);
void dl_sgd_step(DLSGD* opt);

/* Adam / AdamW optimizer */
typedef struct {
    DLParamList* params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    bool adamw;         /* true for decoupled weight decay */
    DLTensor** m;       /* first moment */
    DLTensor** v;       /* second moment */
    int t;              /* timestep */
    DLLRScheduler* scheduler;
} DLAdam;

DLAdam* dl_adam_create(DLParamList* params, float lr, float beta1, float beta2,
                        float eps, float weight_decay, bool adamw);
void dl_adam_free(DLAdam* opt);
void dl_adam_step(DLAdam* opt);
void dl_adam_set_scheduler(DLAdam* opt, DLLRScheduler* sched);

/* Gradient clipping */
float dl_grad_clip_norm(DLParamList* params, float max_norm);

#endif /* DL_OPTIMIZER_H */
