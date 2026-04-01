/*
 * train_gpt2.c - Pre-train a small GPT-2 style model on text data
 *
 * Usage: ./train_gpt2 [text_file] [options]
 *   If no text_file is given, uses a built-in sample text.
 *
 * Options:
 *   --steps N       Number of training steps (default: 200)
 *   --lr F          Learning rate (default: 3e-4)
 *   --batch N       Batch size (default: 4)
 *   --seq N         Sequence length (default: 64)
 *   --layers N      Number of transformer layers (default: 4)
 *   --heads N       Number of attention heads (default: 4)
 *   --dim N         Model dimension (default: 128)
 *   --save PATH     Save model checkpoint path
 */

#include "dl_serialize.h"
#include "dl_dataloader.h"

static const char* SAMPLE_TEXT =
    "The quick brown fox jumps over the lazy dog. "
    "A journey of a thousand miles begins with a single step. "
    "To be or not to be, that is the question. "
    "All that glitters is not gold. "
    "The only thing we have to fear is fear itself. "
    "In the beginning was the word, and the word was with God. "
    "It was the best of times, it was the worst of times. "
    "Call me Ishmael. Some years ago, never mind how long precisely. "
    "It is a truth universally acknowledged that a single man in possession "
    "of a good fortune must be in want of a wife. "
    "All happy families are alike; each unhappy family is unhappy in its own way. "
    "The sun also rises, and the sun goes down. "
    "There is nothing either good or bad, but thinking makes it so. "
    "Knowledge is power. Time is money. Practice makes perfect. "
    "Where there is a will, there is a way. "
    "Actions speak louder than words. The pen is mightier than the sword. "
    "Early to bed and early to rise makes a man healthy, wealthy, and wise. "
    "A picture is worth a thousand words. When in Rome, do as the Romans do. "
    "The best time to plant a tree was twenty years ago. The second best time is now. "
    "Life is what happens when you are busy making other plans. "
    "The purpose of our lives is to be happy. "
    "Get busy living or get busy dying. "
    "You only live once, but if you do it right, once is enough. "
    "Many of life's failures are people who did not realize how close they were "
    "to success when they gave up. "
    "If you want to live a happy life, tie it to a goal, not to people or things. "
    "Never let the fear of striking out keep you from playing the game. "
    "Money and success don't change people; they merely amplify what is already there. "
    "Your time is limited, so don't waste it living someone else's life. "
    "Not how long, but how well you have lived is the main thing. ";

int main(int argc, char** argv) {
    /* Default hyperparameters */
    int n_steps = 200;
    float lr = 3e-4f;
    int batch_size = 4;
    int seq_len = 64;
    int n_layers = 4;
    int n_heads = 4;
    int d_model = 128;
    const char* text_file = NULL;
    const char* save_path = NULL;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = atof(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) seq_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) n_layers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) n_heads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) d_model = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) save_path = argv[++i];
        else if (argv[i][0] != '-') text_file = argv[i];
    }

    printf("=== DeepLearn-C GPT-2 Pre-training ===\n");
    printf("Config: layers=%d, heads=%d, d_model=%d, d_ff=%d\n",
           n_layers, n_heads, d_model, d_model * 4);
    printf("Training: steps=%d, lr=%.1e, batch=%d, seq_len=%d\n",
           n_steps, lr, batch_size, seq_len);

    /* Initialize RNG */
    dl_rng_init(42);

    /* Load or use sample text */
    const char* text = SAMPLE_TEXT;
    char* file_text = NULL;
    if (text_file) {
        FILE* f = fopen(text_file, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long fsize = ftell(f);
            fseek(f, 0, SEEK_SET);
            file_text = (char*)malloc(fsize + 1);
            fread(file_text, 1, fsize, f);
            file_text[fsize] = '\0';
            fclose(f);
            text = file_text;
            printf("Loaded text file: %s (%ld bytes)\n", text_file, fsize);
        } else {
            printf("Warning: cannot open %s, using sample text\n", text_file);
        }
    } else {
        printf("Using built-in sample text (%zu bytes)\n", strlen(text));
    }

    /* Create tokenizer */
    printf("Building tokenizer...\n");
    DLTokenizer* tok = dl_tokenizer_create_char(text);
    printf("Vocab size: %d (character-level)\n", tok->vocab_size);

    /* Optionally train BPE */
    if (strlen(text) > 1000) {
        printf("Training BPE merges...\n");
        dl_tokenizer_train_bpe(tok, text, 100);
        printf("Vocab size after BPE: %d\n", tok->vocab_size);
    }

    /* Create data loader */
    DLDataLoader* dataloader = dl_dataloader_from_text(text, tok, batch_size, seq_len);
    printf("Data: %d tokens, %d batches per epoch\n",
           dataloader->total_tokens, dataloader->n_batches);

    /* Create model */
    DLTransformerConfig config = {
        .vocab_size = tok->vocab_size,
        .max_seq_len = seq_len + 1,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .d_model = d_model,
        .d_ff = d_model * 4,
        .dropout_p = 0.1f,
        .layer_norm_eps = 1e-5f
    };

    printf("Creating model...\n");
    DLTransformerModel* model = dl_transformer_create(config);
    int total_params = dl_paramlist_total_params(model->params);
    printf("Total parameters: %d (%.2f MB)\n", total_params,
           total_params * sizeof(float) / (1024.0 * 1024.0));

    /* Create optimizer */
    DLAdam* optimizer = dl_adam_create(model->params, lr, 0.9f, 0.999f, 1e-8f, 0.01f, true);
    DLLRScheduler* scheduler = dl_scheduler_create(lr, n_steps / 10, n_steps);
    dl_adam_set_scheduler(optimizer, scheduler);

    /* Training loop */
    printf("\n--- Training ---\n");
    dl_graph_init();
    double total_time = 0;
    float running_loss = 0;
    int log_interval = 10;

    for (int step = 0; step < n_steps; step++) {
        /* Get batch */
        if (dl_dataloader_epoch_done(dataloader)) {
            dl_dataloader_reset(dataloader);
            dl_dataloader_shuffle(dataloader);
        }
        const int* batch_tokens = dl_dataloader_next_batch(dataloader);
        if (!batch_tokens) {
            dl_dataloader_reset(dataloader);
            batch_tokens = dl_dataloader_next_batch(dataloader);
        }

        double t0 = dl_time_ms();

        /* Zero gradients */
        dl_paramlist_zero_grad(model->params);
        dl_graph_clear();

        /* Forward + loss */
        dl_transformer_set_training(model, true);
        DLTensor* loss = dl_transformer_loss(model, batch_tokens, batch_size, seq_len);
        float loss_val = loss->data[0];
        running_loss += loss_val;

        /* Backward */
        dl_backward(loss);

        /* Gradient clipping */
        float grad_norm = dl_grad_clip_norm(model->params, 1.0f);

        /* Optimizer step */
        dl_adam_step(optimizer);

        double t1 = dl_time_ms();
        double step_time = t1 - t0;
        total_time += step_time;

        dl_tensor_free(loss);

        /* Logging */
        if ((step + 1) % log_interval == 0 || step == 0) {
            float avg_loss = running_loss / (step < log_interval ? step + 1 : log_interval);
            float current_lr = optimizer->scheduler ?
                dl_scheduler_get_lr(optimizer->scheduler) : optimizer->lr;
            int tokens_per_sec = (int)(batch_size * seq_len * 1000.0 / step_time);

            printf("step %4d/%d | loss: %.4f | lr: %.2e | grad_norm: %.2f | "
                   "%.0fms/step | %d tok/s\n",
                   step + 1, n_steps, avg_loss, current_lr, grad_norm,
                   step_time, tokens_per_sec);

            if ((step + 1) % log_interval == 0) running_loss = 0;
        }
    }

    printf("\nTraining complete! Average time: %.1f ms/step\n", total_time / n_steps);

    /* Generate some text */
    printf("\n--- Generation ---\n");
    dl_transformer_set_training(model, false);

    int prompt[] = {4, 5, 6, 7, 8}; /* First few character tokens */
    int prompt_len = 5;
    int gen_len = 100;

    int* generated = DL_ALLOC(int, prompt_len + gen_len);
    memcpy(generated, prompt, prompt_len * sizeof(int));

    for (int i = 0; i < gen_len; i++) {
        int ctx_len = prompt_len + i;
        if (ctx_len > seq_len) ctx_len = seq_len;
        int* ctx = generated + (prompt_len + i - ctx_len);

        dl_graph_clear();
        int next = dl_transformer_generate_next(model, ctx, ctx_len, 0.8f, 40);
        generated[prompt_len + i] = next;
    }

    char* text_out = dl_tokenizer_decode(tok, generated, prompt_len + gen_len);
    printf("Generated: %s\n", text_out);
    free(text_out);
    free(generated);

    /* Save model */
    if (save_path) {
        printf("\nSaving model to %s...\n", save_path);
        dl_save_model(model, save_path);
        printf("Done!\n");
    }

    /* Cleanup */
    dl_graph_clear();
    dl_transformer_free(model);
    dl_adam_free(optimizer);
    dl_dataloader_free(dataloader);
    dl_tokenizer_free(tok);
    if (file_text) free(file_text);

    printf("\nAll done.\n");
    return 0;
}
