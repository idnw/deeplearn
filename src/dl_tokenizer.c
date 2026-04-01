#include "dl_tokenizer.h"

static char* dl_strdup(const char* s) {
    size_t len = strlen(s);
    char* d = (char*)dl_malloc(len + 1);
    memcpy(d, s, len + 1);
    return d;
}

static void dl_tokenizer_add_token(DLTokenizer* tok, const char* token) {
    if (tok->vocab_size >= tok->capacity) {
        tok->capacity *= 2;
        tok->vocab = (char**)dl_realloc(tok->vocab, tok->capacity * sizeof(char*));
    }
    tok->vocab[tok->vocab_size++] = dl_strdup(token);
}

DLTokenizer* dl_tokenizer_create_char(const char* text) {
    DLTokenizer* tok = DL_ALLOC(DLTokenizer, 1);
    tok->capacity = 512;
    tok->vocab = DL_ALLOC(char*, tok->capacity);
    tok->vocab_size = 0;
    tok->merge_a = NULL;
    tok->merge_b = NULL;
    tok->n_merges = 0;

    /* Build character-level vocab from text */
    bool seen[256] = {false};

    /* First add special tokens */
    dl_tokenizer_add_token(tok, "<pad>");  /* 0 */
    dl_tokenizer_add_token(tok, "<unk>");  /* 1 */
    dl_tokenizer_add_token(tok, "<bos>");  /* 2 */
    dl_tokenizer_add_token(tok, "<eos>");  /* 3 */

    /* Add characters from text */
    for (const char* p = text; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!seen[c]) {
            seen[c] = true;
            char s[2] = {(char)c, '\0'};
            dl_tokenizer_add_token(tok, s);
        }
    }
    return tok;
}

DLTokenizer* dl_tokenizer_load(const char* vocab_path) {
    FILE* f = fopen(vocab_path, "r");
    if (!f) return NULL;

    DLTokenizer* tok = DL_ALLOC(DLTokenizer, 1);
    tok->capacity = 512;
    tok->vocab = DL_ALLOC(char*, tok->capacity);
    tok->vocab_size = 0;
    tok->merge_a = NULL;
    tok->merge_b = NULL;
    tok->n_merges = 0;

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        /* Remove newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len > 0) {
            dl_tokenizer_add_token(tok, line);
        }
    }
    fclose(f);
    return tok;
}

void dl_tokenizer_save(DLTokenizer* tok, const char* vocab_path) {
    FILE* f = fopen(vocab_path, "w");
    DL_CHECK(f != NULL, "cannot open vocab file for writing");
    for (int i = 0; i < tok->vocab_size; i++) {
        fprintf(f, "%s\n", tok->vocab[i]);
    }
    fclose(f);
}

/* BPE training */
void dl_tokenizer_train_bpe(DLTokenizer* tok, const char* text, int n_merges) {
    int text_len = (int)strlen(text);
    if (text_len < 2) return;

    /* Convert text to token ids */
    int* ids = DL_ALLOC(int, text_len);
    int ids_len = 0;

    for (int i = 0; i < text_len; i++) {
        /* Find single character token */
        char s[2] = {text[i], '\0'};
        int found = 1; /* default to <unk> */
        for (int v = 0; v < tok->vocab_size; v++) {
            if (strcmp(tok->vocab[v], s) == 0) {
                found = v;
                break;
            }
        }
        ids[ids_len++] = found;
    }

    /* Allocate merge arrays */
    tok->merge_a = DL_ALLOC(int, n_merges);
    tok->merge_b = DL_ALLOC(int, n_merges);
    tok->n_merges = 0;

    for (int merge = 0; merge < n_merges && ids_len >= 2; merge++) {
        /* Count all pairs */
        int max_count = 0;
        int best_a = -1, best_b = -1;

        /* Simple O(n*v^2) approach - sufficient for small vocabs */
        for (int i = 0; i < ids_len - 1; i++) {
            int a = ids[i], b = ids[i + 1];
            int count = 0;
            for (int j = i; j < ids_len - 1; j++) {
                if (ids[j] == a && ids[j + 1] == b) count++;
            }
            if (count > max_count) {
                max_count = count;
                best_a = a;
                best_b = b;
            }
        }

        if (max_count < 2) break; /* No frequent pairs left */

        /* Create merged token */
        char merged[256];
        snprintf(merged, sizeof(merged), "%s%s", tok->vocab[best_a], tok->vocab[best_b]);
        dl_tokenizer_add_token(tok, merged);
        int new_id = tok->vocab_size - 1;

        tok->merge_a[tok->n_merges] = best_a;
        tok->merge_b[tok->n_merges] = best_b;
        tok->n_merges++;

        /* Apply merge to ids */
        int new_len = 0;
        for (int i = 0; i < ids_len; i++) {
            if (i < ids_len - 1 && ids[i] == best_a && ids[i + 1] == best_b) {
                ids[new_len++] = new_id;
                i++; /* skip next */
            } else {
                ids[new_len++] = ids[i];
            }
        }
        ids_len = new_len;
    }

    free(ids);
}

int* dl_tokenizer_encode(DLTokenizer* tok, const char* text, int* out_len) {
    int text_len = (int)strlen(text);
    int* ids = DL_ALLOC(int, text_len + 1);
    int ids_len = 0;

    /* Character-level encoding */
    for (int i = 0; i < text_len; i++) {
        char s[2] = {text[i], '\0'};
        int found = 1; /* <unk> */
        for (int v = 0; v < tok->vocab_size; v++) {
            if (strcmp(tok->vocab[v], s) == 0) {
                found = v;
                break;
            }
        }
        ids[ids_len++] = found;
    }

    /* Apply BPE merges in order */
    for (int m = 0; m < tok->n_merges; m++) {
        int a = tok->merge_a[m];
        int b = tok->merge_b[m];
        int new_id = m + (tok->vocab_size - tok->n_merges); /* merge tokens start after base vocab */

        int new_len = 0;
        for (int i = 0; i < ids_len; i++) {
            if (i < ids_len - 1 && ids[i] == a && ids[i + 1] == b) {
                ids[new_len++] = new_id;
                i++;
            } else {
                ids[new_len++] = ids[i];
            }
        }
        ids_len = new_len;
    }

    *out_len = ids_len;
    return ids;
}

char* dl_tokenizer_decode(DLTokenizer* tok, const int* tokens, int n_tokens) {
    /* Estimate output size */
    size_t total_len = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            total_len += strlen(tok->vocab[tokens[i]]);
        }
    }

    char* out = (char*)dl_malloc(total_len + 1);
    out[0] = '\0';

    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
            strcat(out, tok->vocab[tokens[i]]);
        }
    }
    return out;
}

void dl_tokenizer_free(DLTokenizer* tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->vocab[i]);
    }
    free(tok->vocab);
    if (tok->merge_a) free(tok->merge_a);
    if (tok->merge_b) free(tok->merge_b);
    free(tok);
}
