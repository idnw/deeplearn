/*
 * inference.c - Load a trained model and generate text
 *
 * Usage: ./inference <model_path> [options]
 *
 * Options:
 *   --prompt TEXT    Prompt text (default: "The ")
 *   --length N      Number of tokens to generate (default: 200)
 *   --temp F        Temperature (default: 0.8)
 *   --topk N        Top-k sampling (default: 40)
 *   --vocab PATH    Vocabulary file path
 */

#include "dl_serialize.h"
#include "dl_tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [--prompt TEXT] [--length N] "
               "[--temp F] [--topk N] [--vocab PATH]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt_text = "The ";
    const char* vocab_path = NULL;
    int gen_length = 200;
    float temperature = 0.8f;
    int top_k = 40;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt_text = argv[++i];
        else if (strcmp(argv[i], "--length") == 0 && i + 1 < argc) gen_length = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temperature = atof(argv[++i]);
        else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab_path = argv[++i];
    }

    dl_rng_init(42);

    /* Load or create tokenizer */
    DLTokenizer* tok = NULL;
    if (vocab_path) {
        tok = dl_tokenizer_load(vocab_path);
        DL_CHECK(tok != NULL, "Failed to load vocabulary");
    } else {
        /* Create a basic ASCII tokenizer */
        printf("No vocab file specified, creating basic ASCII tokenizer\n");
        /* We need the original tokenizer - for demo, create char-level */
        const char* ascii = " !\"#$%&'()*+,-./0123456789:;<=>?@"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                            "abcdefghijklmnopqrstuvwxyz{|}~\n\t";
        tok = dl_tokenizer_create_char(ascii);
    }
    printf("Vocab size: %d\n", tok->vocab_size);

    /* Create model with default config (must match training config) */
    DLTransformerConfig config = {
        .vocab_size = tok->vocab_size,
        .max_seq_len = 256,
        .n_layers = 4,
        .n_heads = 4,
        .d_model = 128,
        .d_ff = 512,
        .dropout_p = 0.0f,
        .layer_norm_eps = 1e-5f
    };

    DLTransformerModel* model = dl_transformer_create(config);
    dl_transformer_set_training(model, false);

    /* Load weights */
    printf("Loading model from %s...\n", model_path);
    int ret = dl_load_model(model, model_path);
    if (ret != 0) {
        printf("Error loading model (code %d)\n", ret);
        dl_transformer_free(model);
        dl_tokenizer_free(tok);
        return 1;
    }
    printf("Model loaded successfully (%d params)\n",
           dl_paramlist_total_params(model->params));

    /* Encode prompt */
    int prompt_len;
    int* prompt_tokens = dl_tokenizer_encode(tok, prompt_text, &prompt_len);
    printf("Prompt (%d tokens): %s\n", prompt_len, prompt_text);

    /* Generate */
    printf("\n--- Generation ---\n");
    int max_len = prompt_len + gen_length;
    int* tokens = DL_ALLOC(int, max_len);
    memcpy(tokens, prompt_tokens, prompt_len * sizeof(int));

    printf("%s", prompt_text);
    fflush(stdout);

    dl_graph_init();
    for (int i = 0; i < gen_length; i++) {
        int ctx_len = prompt_len + i;
        if (ctx_len > (int)config.max_seq_len) {
            ctx_len = config.max_seq_len;
        }
        int* ctx_start = tokens + (prompt_len + i - ctx_len);

        dl_graph_clear();
        int next = dl_transformer_generate_next(model, ctx_start, ctx_len,
                                                 temperature, top_k);
        tokens[prompt_len + i] = next;

        /* Print token */
        if (next >= 0 && next < tok->vocab_size) {
            printf("%s", tok->vocab[next]);
            fflush(stdout);
        }
    }
    printf("\n\n");

    /* Cleanup */
    free(tokens);
    free(prompt_tokens);
    dl_graph_clear();
    dl_transformer_free(model);
    dl_tokenizer_free(tok);

    return 0;
}
