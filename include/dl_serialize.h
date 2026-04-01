#ifndef DL_SERIALIZE_H
#define DL_SERIALIZE_H

#include "dl_transformer.h"
#include "dl_optimizer.h"

/* ========== Custom Binary Format ========== */
/* Format: MAGIC(4) VERSION(4) N_TENSORS(4) [TENSOR_ENTRIES...] [DATA...]
 * Each entry: NAME_LEN(4) NAME NDIM(4) SHAPE[NDIM] OFFSET(8) SIZE(8)
 */

#define DL_MAGIC 0x444C4D4C  /* "DLML" */
#define DL_VERSION 1

/* Save model parameters to custom format */
int dl_save_model(DLTransformerModel* model, const char* path);

/* Load model parameters from custom format */
int dl_load_model(DLTransformerModel* model, const char* path);

/* Save/load optimizer state */
int dl_save_checkpoint(DLTransformerModel* model, DLAdam* opt, int step, const char* path);
int dl_load_checkpoint(DLTransformerModel* model, DLAdam* opt, int* step, const char* path);

/* ========== GGUF Format Reader ========== */

/* GGUF tensor types */
typedef enum {
    GGUF_TYPE_F32  = 0,
    GGUF_TYPE_F16  = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q8_0 = 8,
} GGUFType;

typedef struct {
    char* name;
    int ndim;
    int64_t shape[DL_MAX_DIMS];
    GGUFType type;
    uint64_t offset;
    float* data;     /* converted to f32 */
    int size;        /* total elements */
} GGUFTensor;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_metadata;

    GGUFTensor* tensors;
    uint64_t data_offset;
} GGUFFile;

/* Load GGUF file and parse all tensors */
GGUFFile* dl_gguf_load(const char* path);

/* Find a tensor by name */
GGUFTensor* dl_gguf_find_tensor(GGUFFile* gguf, const char* name);

/* Load GGUF weights into a transformer model (best-effort name matching) */
int dl_gguf_load_into_model(GGUFFile* gguf, DLTransformerModel* model);

void dl_gguf_free(GGUFFile* gguf);

#endif /* DL_SERIALIZE_H */
