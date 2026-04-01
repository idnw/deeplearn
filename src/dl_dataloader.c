#include "dl_dataloader.h"

static DLDataLoader* dl_dataloader_from_tokens(int* tokens, int total_tokens,
                                                 int batch_size, int seq_len) {
    DLDataLoader* dl = DL_ALLOC(DLDataLoader, 1);
    dl->tokens = tokens;
    dl->total_tokens = total_tokens;
    dl->seq_len = seq_len;
    dl->batch_size = batch_size;

    int n_samples = (total_tokens - 1) / seq_len;
    dl->n_batches = n_samples / batch_size;
    if (dl->n_batches < 1) dl->n_batches = 1;

    dl->current_batch = 0;

    /* Initialize sequential batch order */
    dl->shuffle_indices = DL_ALLOC(int, dl->n_batches);
    for (int i = 0; i < dl->n_batches; i++) {
        dl->shuffle_indices[i] = i;
    }
    return dl;
}

DLDataLoader* dl_dataloader_create(const char* filepath, DLTokenizer* tok,
                                    int batch_size, int seq_len) {
    FILE* f = fopen(filepath, "r");
    DL_CHECK(f != NULL, "cannot open data file");

    /* Read entire file */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* text = (char*)dl_malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);

    DLDataLoader* dl = dl_dataloader_from_text(text, tok, batch_size, seq_len);
    free(text);
    return dl;
}

DLDataLoader* dl_dataloader_from_text(const char* text, DLTokenizer* tok,
                                       int batch_size, int seq_len) {
    int n_tokens;
    int* tokens = dl_tokenizer_encode(tok, text, &n_tokens);
    return dl_dataloader_from_tokens(tokens, n_tokens, batch_size, seq_len);
}

const int* dl_dataloader_next_batch(DLDataLoader* dl) {
    if (dl->current_batch >= dl->n_batches) return NULL;

    int batch_idx = dl->shuffle_indices[dl->current_batch];
    int offset = batch_idx * dl->batch_size * dl->seq_len;

    /* Ensure we don't go out of bounds */
    if (offset + dl->batch_size * (dl->seq_len + 1) > dl->total_tokens) {
        offset = dl->total_tokens - dl->batch_size * (dl->seq_len + 1);
        if (offset < 0) offset = 0;
    }

    dl->current_batch++;
    return dl->tokens + offset;
}

void dl_dataloader_shuffle(DLDataLoader* dl) {
    dl_rng_ensure_init();
    for (int i = dl->n_batches - 1; i > 0; i--) {
        int j = (int)(dl_rng_float(&dl_global_rng) * (i + 1));
        if (j > i) j = i;
        int tmp = dl->shuffle_indices[i];
        dl->shuffle_indices[i] = dl->shuffle_indices[j];
        dl->shuffle_indices[j] = tmp;
    }
}

void dl_dataloader_reset(DLDataLoader* dl) {
    dl->current_batch = 0;
}

bool dl_dataloader_epoch_done(DLDataLoader* dl) {
    return dl->current_batch >= dl->n_batches;
}

void dl_dataloader_free(DLDataLoader* dl) {
    if (!dl) return;
    free(dl->tokens);
    free(dl->shuffle_indices);
    free(dl);
}
