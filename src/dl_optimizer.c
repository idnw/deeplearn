#include "dl_optimizer.h"

/* ========== Learning Rate Scheduler (Warmup + Cosine Decay) ========== */

DLLRScheduler* dl_scheduler_create(float base_lr, int warmup_steps, int total_steps) {
    DLLRScheduler* sched = DL_ALLOC(DLLRScheduler, 1);
    sched->base_lr = base_lr;
    sched->warmup_steps = warmup_steps;
    sched->total_steps = total_steps;
    sched->current_step = 0;
    return sched;
}

void dl_scheduler_free(DLLRScheduler* sched) {
    if (sched) free(sched);
}

float dl_scheduler_get_lr(DLLRScheduler* sched) {
    int step = sched->current_step;
    if (step < sched->warmup_steps) {
        /* Linear warmup */
        return sched->base_lr * (float)(step + 1) / (float)sched->warmup_steps;
    } else {
        /* Cosine decay */
        float progress = (float)(step - sched->warmup_steps) /
                         (float)(sched->total_steps - sched->warmup_steps);
        if (progress > 1.0f) progress = 1.0f;
        return sched->base_lr * 0.5f * (1.0f + cosf(3.14159265358979f * progress));
    }
}

void dl_scheduler_step(DLLRScheduler* sched) {
    sched->current_step++;
}

/* ========== SGD ========== */

DLSGD* dl_sgd_create(DLParamList* params, float lr, float momentum, float weight_decay) {
    DLSGD* opt = DL_ALLOC(DLSGD, 1);
    opt->params = params;
    opt->lr = lr;
    opt->momentum = momentum;
    opt->weight_decay = weight_decay;

    if (momentum > 0) {
        opt->velocity = DL_ALLOC(DLTensor*, params->n_params);
        for (int i = 0; i < params->n_params; i++) {
            opt->velocity[i] = dl_tensor_zeros(params->params[i]->shape,
                                                params->params[i]->ndim);
        }
    } else {
        opt->velocity = NULL;
    }
    return opt;
}

void dl_sgd_free(DLSGD* opt) {
    if (!opt) return;
    if (opt->velocity) {
        for (int i = 0; i < opt->params->n_params; i++) {
            dl_tensor_free(opt->velocity[i]);
        }
        free(opt->velocity);
    }
    free(opt);
}

void dl_sgd_step(DLSGD* opt) {
    for (int i = 0; i < opt->params->n_params; i++) {
        DLTensor* p = opt->params->params[i];
        DLTensor* g = p->grad;
        if (!g) continue;

        /* Weight decay */
        if (opt->weight_decay > 0) {
            for (int j = 0; j < p->size; j++) {
                g->data[j] += opt->weight_decay * p->data[j];
            }
        }

        if (opt->momentum > 0) {
            DLTensor* v = opt->velocity[i];
            for (int j = 0; j < p->size; j++) {
                v->data[j] = opt->momentum * v->data[j] + g->data[j];
                p->data[j] -= opt->lr * v->data[j];
            }
        } else {
            for (int j = 0; j < p->size; j++) {
                p->data[j] -= opt->lr * g->data[j];
            }
        }
    }
}

/* ========== Adam / AdamW ========== */

DLAdam* dl_adam_create(DLParamList* params, float lr, float beta1, float beta2,
                        float eps, float weight_decay, bool adamw) {
    DLAdam* opt = DL_ALLOC(DLAdam, 1);
    opt->params = params;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->adamw = adamw;
    opt->t = 0;
    opt->scheduler = NULL;

    opt->m = DL_ALLOC(DLTensor*, params->n_params);
    opt->v = DL_ALLOC(DLTensor*, params->n_params);
    for (int i = 0; i < params->n_params; i++) {
        opt->m[i] = dl_tensor_zeros(params->params[i]->shape, params->params[i]->ndim);
        opt->v[i] = dl_tensor_zeros(params->params[i]->shape, params->params[i]->ndim);
    }
    return opt;
}

void dl_adam_free(DLAdam* opt) {
    if (!opt) return;
    for (int i = 0; i < opt->params->n_params; i++) {
        dl_tensor_free(opt->m[i]);
        dl_tensor_free(opt->v[i]);
    }
    free(opt->m);
    free(opt->v);
    if (opt->scheduler) dl_scheduler_free(opt->scheduler);
    free(opt);
}

void dl_adam_set_scheduler(DLAdam* opt, DLLRScheduler* sched) {
    opt->scheduler = sched;
}

void dl_adam_step(DLAdam* opt) {
    opt->t++;
    float lr = opt->scheduler ? dl_scheduler_get_lr(opt->scheduler) : opt->lr;
    if (opt->scheduler) dl_scheduler_step(opt->scheduler);

    float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
    float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);

    for (int i = 0; i < opt->params->n_params; i++) {
        DLTensor* p = opt->params->params[i];
        DLTensor* g = p->grad;
        if (!g) continue;

        DLTensor* m = opt->m[i];
        DLTensor* v = opt->v[i];

        for (int j = 0; j < p->size; j++) {
            float grad = g->data[j];

            /* L2 regularization (classic Adam) */
            if (!opt->adamw && opt->weight_decay > 0) {
                grad += opt->weight_decay * p->data[j];
            }

            /* Update biased first and second moment estimates */
            m->data[j] = opt->beta1 * m->data[j] + (1.0f - opt->beta1) * grad;
            v->data[j] = opt->beta2 * v->data[j] + (1.0f - opt->beta2) * grad * grad;

            /* Bias-corrected estimates */
            float m_hat = m->data[j] / bc1;
            float v_hat = v->data[j] / bc2;

            /* Update parameter */
            float update = lr * m_hat / (sqrtf(v_hat) + opt->eps);

            /* AdamW: decoupled weight decay */
            if (opt->adamw && opt->weight_decay > 0) {
                update += lr * opt->weight_decay * p->data[j];
            }

            p->data[j] -= update;
        }
    }
}

/* ========== Gradient Clipping ========== */

float dl_grad_clip_norm(DLParamList* params, float max_norm) {
    /* Compute total gradient norm */
    float total_norm_sq = 0.0f;
    for (int i = 0; i < params->n_params; i++) {
        DLTensor* g = params->params[i]->grad;
        if (!g) continue;
        for (int j = 0; j < g->size; j++) {
            total_norm_sq += g->data[j] * g->data[j];
        }
    }
    float total_norm = sqrtf(total_norm_sq);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (int i = 0; i < params->n_params; i++) {
            DLTensor* g = params->params[i]->grad;
            if (!g) continue;
            dl_tensor_scale_(g, scale);
        }
    }
    return total_norm;
}
