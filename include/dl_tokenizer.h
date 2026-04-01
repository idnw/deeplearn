#ifndef DL_TOKENIZER_H
#define DL_TOKENIZER_H

#include "dl_common.h"

/* Simple character-level + BPE tokenizer */
typedef struct {
    char** vocab;       /* token strings */
    int vocab_size;
    int capacity;

    /* BPE merge rules: merge_a[i] + merge_b[i] -> i + base_vocab_size */
    int* merge_a;
    int* merge_b;
    int n_merges;
} DLTokenizer;

/* Create a character-level tokenizer from text */
DLTokenizer* dl_tokenizer_create_char(const char* text);

/* Create from vocab file (one token per line) */
DLTokenizer* dl_tokenizer_load(const char* vocab_path);

/* Save vocab to file */
void dl_tokenizer_save(DLTokenizer* tok, const char* vocab_path);

/* Train BPE merges on text */
void dl_tokenizer_train_bpe(DLTokenizer* tok, const char* text, int n_merges);

/* Encode text to token ids */
int* dl_tokenizer_encode(DLTokenizer* tok, const char* text, int* out_len);

/* Decode token ids to text */
char* dl_tokenizer_decode(DLTokenizer* tok, const int* tokens, int n_tokens);

void dl_tokenizer_free(DLTokenizer* tok);

#endif /* DL_TOKENIZER_H */
