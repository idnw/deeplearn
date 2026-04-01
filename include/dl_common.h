#ifndef DL_COMMON_H
#define DL_COMMON_H

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <float.h>

/* SIMD detection */
#ifdef __AVX2__
    #include <immintrin.h>
    #define DL_USE_AVX2 1
#elif defined(__SSE2__)
    #include <emmintrin.h>
    #define DL_USE_SSE2 1
#endif

#define DL_MAX_DIMS 8
#define DL_MAX_PARAMS 4096
#define DL_MAX_GRAPH_NODES 262144

#define DL_CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "DL ERROR [%s:%d]: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while(0)

#define DL_ALLOC(type, count) ((type*)dl_malloc(sizeof(type) * (count)))
#define DL_FREE(ptr) do { free(ptr); (ptr) = NULL; } while(0)

static inline void* dl_malloc(size_t size) {
    void* ptr = malloc(size);
    DL_CHECK(ptr != NULL, "out of memory");
    return ptr;
}

static inline void* dl_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    DL_CHECK(ptr != NULL, "out of memory");
    return ptr;
}

static inline void* dl_realloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    DL_CHECK(new_ptr != NULL, "out of memory");
    return new_ptr;
}

/* Arena allocator for computational graph memory */
typedef struct {
    char* base;
    size_t used;
    size_t capacity;
} DLArena;

static inline DLArena* dl_arena_create(size_t capacity) {
    DLArena* arena = DL_ALLOC(DLArena, 1);
    arena->base = (char*)dl_malloc(capacity);
    arena->used = 0;
    arena->capacity = capacity;
    return arena;
}

static inline void* dl_arena_alloc(DLArena* arena, size_t size) {
    /* Align to 32 bytes for SIMD */
    size_t aligned = (size + 31) & ~31;
    DL_CHECK(arena->used + aligned <= arena->capacity, "arena out of memory");
    void* ptr = arena->base + arena->used;
    arena->used += aligned;
    return ptr;
}

static inline void dl_arena_reset(DLArena* arena) {
    arena->used = 0;
}

static inline void dl_arena_free(DLArena* arena) {
    if (arena) {
        free(arena->base);
        free(arena);
    }
}

/* Random number generation - xoshiro256** */
typedef struct {
    uint64_t s[4];
} DLRng;

static inline uint64_t dl_rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t dl_rng_next(DLRng* rng) {
    uint64_t result = dl_rng_rotl(rng->s[1] * 5, 7) * 9;
    uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = dl_rng_rotl(rng->s[3], 45);
    return result;
}

static inline float dl_rng_float(DLRng* rng) {
    return (dl_rng_next(rng) >> 11) * (1.0f / 9007199254740992.0f);
}

static inline float dl_rng_normal(DLRng* rng) {
    /* Box-Muller transform */
    float u1 = dl_rng_float(rng);
    float u2 = dl_rng_float(rng);
    while (u1 < 1e-10f) u1 = dl_rng_float(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

static inline DLRng dl_rng_seed(uint64_t seed) {
    DLRng rng;
    /* SplitMix64 for seeding */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng.s[i] = z ^ (z >> 31);
    }
    return rng;
}

/* Global RNG */
static DLRng dl_global_rng;
static bool dl_rng_initialized = false;

static inline void dl_rng_init(uint64_t seed) {
    dl_global_rng = dl_rng_seed(seed);
    dl_rng_initialized = true;
}

static inline void dl_rng_ensure_init(void) {
    if (!dl_rng_initialized) {
        dl_rng_init((uint64_t)time(NULL));
    }
}

/* Timer utility */
static inline double dl_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#endif /* DL_COMMON_H */
