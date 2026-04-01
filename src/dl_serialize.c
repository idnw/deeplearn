#include "dl_serialize.h"

/* ========== Custom Binary Format ========== */

typedef struct {
    char name[128];
    int ndim;
    int shape[DL_MAX_DIMS];
    uint64_t data_offset;
    uint64_t data_size;
} DLTensorEntry;

static void dl_write_param(const char* name, DLTensor* t,
                            DLTensorEntry* entry, uint64_t* offset) {
    strncpy(entry->name, name, 127);
    entry->name[127] = '\0';
    entry->ndim = t->ndim;
    memcpy(entry->shape, t->shape, t->ndim * sizeof(int));
    entry->data_offset = *offset;
    entry->data_size = t->size * sizeof(float);
    *offset += entry->data_size;
}

int dl_save_model(DLTransformerModel* model, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    DLParamList* params = model->params;
    int n = params->n_params;

    /* Write header */
    uint32_t magic = DL_MAGIC;
    uint32_t version = DL_VERSION;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n, 4, 1, f);

    /* Write config */
    fwrite(&model->config, sizeof(DLTransformerConfig), 1, f);

    /* Build entries */
    DLTensorEntry* entries = DL_ALLOC(DLTensorEntry, n);
    uint64_t data_offset = 0;
    for (int i = 0; i < n; i++) {
        char name[128];
        snprintf(name, sizeof(name), "param_%d", i);
        dl_write_param(name, params->params[i], &entries[i], &data_offset);
    }

    /* Write entries */
    fwrite(entries, sizeof(DLTensorEntry), n, f);

    /* Write parameter data */
    for (int i = 0; i < n; i++) {
        DLTensor* t = params->params[i];
        DLTensor* ct = dl_tensor_contiguous(t);
        fwrite(ct->data, sizeof(float), ct->size, f);
        dl_tensor_free(ct);
    }

    free(entries);
    fclose(f);
    return 0;
}

int dl_load_model(DLTransformerModel* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version;
    int n;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&n, 4, 1, f);

    if (magic != DL_MAGIC || version != DL_VERSION) {
        fclose(f);
        return -2;
    }

    /* Read config (skip, model already configured) */
    DLTransformerConfig config;
    fread(&config, sizeof(DLTransformerConfig), 1, f);

    DL_CHECK(n == model->params->n_params, "param count mismatch");

    /* Read entries */
    DLTensorEntry* entries = DL_ALLOC(DLTensorEntry, n);
    fread(entries, sizeof(DLTensorEntry), n, f);

    /* Read parameter data */
    for (int i = 0; i < n; i++) {
        DLTensor* t = model->params->params[i];
        DL_CHECK((int)entries[i].data_size == (int)(t->size * sizeof(float)),
                 "param size mismatch");
        fread(t->data, sizeof(float), t->size, f);
    }

    free(entries);
    fclose(f);
    return 0;
}

int dl_save_checkpoint(DLTransformerModel* model, DLAdam* opt, int step, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t magic = DL_MAGIC;
    uint32_t version = DL_VERSION;
    uint32_t checkpoint_flag = 0xCCCC;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&checkpoint_flag, 4, 1, f);
    fwrite(&step, 4, 1, f);
    fwrite(&model->config, sizeof(DLTransformerConfig), 1, f);

    int n = model->params->n_params;
    fwrite(&n, 4, 1, f);

    /* Write model params */
    for (int i = 0; i < n; i++) {
        DLTensor* t = model->params->params[i];
        fwrite(&t->size, 4, 1, f);
        fwrite(t->data, sizeof(float), t->size, f);
    }

    /* Write optimizer state */
    if (opt) {
        int has_opt = 1;
        fwrite(&has_opt, 4, 1, f);
        fwrite(&opt->t, 4, 1, f);
        for (int i = 0; i < n; i++) {
            fwrite(opt->m[i]->data, sizeof(float), opt->m[i]->size, f);
            fwrite(opt->v[i]->data, sizeof(float), opt->v[i]->size, f);
        }
    } else {
        int has_opt = 0;
        fwrite(&has_opt, 4, 1, f);
    }

    fclose(f);
    return 0;
}

int dl_load_checkpoint(DLTransformerModel* model, DLAdam* opt, int* step, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version, checkpoint_flag;
    fread(&magic, 4, 1, f);
    fread(&version, 4, 1, f);
    fread(&checkpoint_flag, 4, 1, f);
    fread(step, 4, 1, f);

    DLTransformerConfig config;
    fread(&config, sizeof(DLTransformerConfig), 1, f);

    int n;
    fread(&n, 4, 1, f);
    DL_CHECK(n == model->params->n_params, "checkpoint param count mismatch");

    for (int i = 0; i < n; i++) {
        int size;
        fread(&size, 4, 1, f);
        DL_CHECK(size == model->params->params[i]->size, "checkpoint param size mismatch");
        fread(model->params->params[i]->data, sizeof(float), size, f);
    }

    int has_opt;
    fread(&has_opt, 4, 1, f);
    if (has_opt && opt) {
        fread(&opt->t, 4, 1, f);
        for (int i = 0; i < n; i++) {
            fread(opt->m[i]->data, sizeof(float), opt->m[i]->size, f);
            fread(opt->v[i]->data, sizeof(float), opt->v[i]->size, f);
        }
    }

    fclose(f);
    return 0;
}

/* ========== GGUF Reader ========== */

/* Half-float to float conversion */
static float dl_f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, 4);
            return f;
        }
        /* Denormal */
        exponent = 1;
        while (!(mantissa & 0x400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x3FF;
        exponent += 127 - 15;
    } else if (exponent == 31) {
        exponent = 255; /* Inf/NaN */
    } else {
        exponent += 127 - 15;
    }

    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, 4);
    return f;
}

/* GGUF string reader */
static char* dl_gguf_read_string(FILE* f) {
    uint64_t len;
    fread(&len, 8, 1, f);
    if (len > 65536) len = 65536; /* sanity check */
    char* s = (char*)dl_malloc(len + 1);
    fread(s, 1, len, f);
    s[len] = '\0';
    return s;
}

/* GGUF metadata value types */
enum {
    GGUF_META_UINT8 = 0, GGUF_META_INT8, GGUF_META_UINT16, GGUF_META_INT16,
    GGUF_META_UINT32, GGUF_META_INT32, GGUF_META_FLOAT32,
    GGUF_META_BOOL, GGUF_META_STRING,
    GGUF_META_ARRAY, GGUF_META_UINT64, GGUF_META_INT64, GGUF_META_FLOAT64
};

static void dl_gguf_skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case GGUF_META_UINT8:  case GGUF_META_INT8:  case GGUF_META_BOOL:
            fseek(f, 1, SEEK_CUR); break;
        case GGUF_META_UINT16: case GGUF_META_INT16:
            fseek(f, 2, SEEK_CUR); break;
        case GGUF_META_UINT32: case GGUF_META_INT32: case GGUF_META_FLOAT32:
            fseek(f, 4, SEEK_CUR); break;
        case GGUF_META_UINT64: case GGUF_META_INT64: case GGUF_META_FLOAT64:
            fseek(f, 8, SEEK_CUR); break;
        case GGUF_META_STRING: {
            char* s = dl_gguf_read_string(f);
            free(s);
            break;
        }
        case GGUF_META_ARRAY: {
            uint32_t arr_type;
            uint64_t arr_len;
            fread(&arr_type, 4, 1, f);
            fread(&arr_len, 8, 1, f);
            for (uint64_t i = 0; i < arr_len; i++) {
                dl_gguf_skip_value(f, arr_type);
            }
            break;
        }
    }
}

GGUFFile* dl_gguf_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    GGUFFile* gguf = DL_ALLOC(GGUFFile, 1);
    memset(gguf, 0, sizeof(GGUFFile));

    /* Read header */
    fread(&gguf->magic, 4, 1, f);
    if (gguf->magic != 0x46475547) { /* "GGUF" */
        fprintf(stderr, "Not a GGUF file\n");
        free(gguf);
        fclose(f);
        return NULL;
    }

    fread(&gguf->version, 4, 1, f);
    fread(&gguf->n_tensors, 8, 1, f);
    fread(&gguf->n_metadata, 8, 1, f);

    /* Skip metadata */
    for (uint64_t i = 0; i < gguf->n_metadata; i++) {
        char* key = dl_gguf_read_string(f);
        uint32_t value_type;
        fread(&value_type, 4, 1, f);
        dl_gguf_skip_value(f, value_type);
        free(key);
    }

    /* Read tensor info */
    gguf->tensors = DL_ALLOC(GGUFTensor, gguf->n_tensors);
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        GGUFTensor* t = &gguf->tensors[i];
        t->name = dl_gguf_read_string(f);

        uint32_t ndim;
        fread(&ndim, 4, 1, f);
        t->ndim = ndim;
        t->size = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            fread(&t->shape[d], 8, 1, f);
            t->size *= (int)t->shape[d];
        }

        uint32_t type;
        fread(&type, 4, 1, f);
        t->type = (GGUFType)type;

        fread(&t->offset, 8, 1, f);
        t->data = NULL;
    }

    /* Alignment - data section starts at 32-byte boundary */
    long pos = ftell(f);
    long aligned = (pos + 31) & ~31;
    gguf->data_offset = aligned;

    /* Load tensor data */
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        GGUFTensor* t = &gguf->tensors[i];
        fseek(f, gguf->data_offset + t->offset, SEEK_SET);

        t->data = DL_ALLOC(float, t->size);

        switch (t->type) {
            case GGUF_TYPE_F32:
                fread(t->data, sizeof(float), t->size, f);
                break;
            case GGUF_TYPE_F16: {
                uint16_t* f16 = DL_ALLOC(uint16_t, t->size);
                fread(f16, sizeof(uint16_t), t->size, f);
                for (int j = 0; j < t->size; j++) {
                    t->data[j] = dl_f16_to_f32(f16[j]);
                }
                free(f16);
                break;
            }
            default:
                fprintf(stderr, "Warning: unsupported GGUF type %d for tensor '%s', skipping\n",
                        t->type, t->name);
                memset(t->data, 0, t->size * sizeof(float));
                break;
        }
    }

    fclose(f);
    return gguf;
}

GGUFTensor* dl_gguf_find_tensor(GGUFFile* gguf, const char* name) {
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }
    return NULL;
}

int dl_gguf_load_into_model(GGUFFile* gguf, DLTransformerModel* model) {
    int loaded = 0;

    /* Try to match GGUF tensor names to model parameters */
    /* Common naming: token_embd.weight, blk.0.attn_q.weight, etc. */

    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        GGUFTensor* gt = &gguf->tensors[i];

        /* Match by checking parameter shapes */
        for (int j = 0; j < model->params->n_params; j++) {
            DLTensor* pt = model->params->params[j];
            if (pt->size == gt->size) {
                /* Size matches - copy data */
                memcpy(pt->data, gt->data, pt->size * sizeof(float));
                loaded++;
                break;
            }
        }
    }

    return loaded;
}

void dl_gguf_free(GGUFFile* gguf) {
    if (!gguf) return;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        free(gguf->tensors[i].name);
        free(gguf->tensors[i].data);
    }
    free(gguf->tensors);
    free(gguf);
}
