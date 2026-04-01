#ifndef DL_DATALOADER_H
#define DL_DATALOADER_H

#include "dl_tokenizer.h"

typedef struct {
    int* tokens;         /* all tokenized data */
    int total_tokens;
    int seq_len;
    int batch_size;
    int n_batches;
    int current_batch;
    int* shuffle_indices; /* batch order */
} DLDataLoader;

/* Create dataloader from text file */
DLDataLoader* dl_dataloader_create(const char* filepath, DLTokenizer* tok,
                                    int batch_size, int seq_len);

/* Create from raw text */
DLDataLoader* dl_dataloader_from_text(const char* text, DLTokenizer* tok,
                                       int batch_size, int seq_len);

/* Get next batch: returns pointer to (batch_size * (seq_len+1)) tokens
 * The +1 is for next-token prediction targets */
const int* dl_dataloader_next_batch(DLDataLoader* dl);

/* Shuffle batch order */
void dl_dataloader_shuffle(DLDataLoader* dl);

/* Reset to beginning */
void dl_dataloader_reset(DLDataLoader* dl);

/* Check if epoch is complete */
bool dl_dataloader_epoch_done(DLDataLoader* dl);

void dl_dataloader_free(DLDataLoader* dl);

#endif /* DL_DATALOADER_H */
