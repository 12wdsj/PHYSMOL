#ifndef PHYSMOL_MEMORY_H
#define PHYSMOL_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Three-tier memory hierarchy:
 * L1 (Hot):  RAM pool for active vectors
 * L2 (Warm): HashMap for codebook / frequently accessed
 * L3 (Cold): File-backed storage for historical data
 */

typedef enum {
    MEM_TIER_L1_HOT = 0,
    MEM_TIER_L2_WARM = 1,
    MEM_TIER_L3_COLD = 2
} MemTier;

/* Simple hashmap entry for L2 */
typedef struct MemEntry {
    uint64_t key;
    float *data;
    size_t size;      /* size in floats */
    struct MemEntry *next;
} MemEntry;

/* L2 hashmap */
typedef struct {
    MemEntry **buckets;
    size_t num_buckets;
    size_t num_entries;
} MemHashMap;

/* Tiered memory manager */
typedef struct {
    /* L1: direct memory pool */
    float **l1_slots;
    size_t *l1_sizes;
    size_t l1_capacity;
    size_t l1_count;

    /* L2: hashmap */
    MemHashMap l2;

    /* L3: file path */
    char *l3_path;

    /* Stats */
    size_t l1_hits;
    size_t l2_hits;
    size_t l3_hits;
    size_t misses;
} TieredMemory;

/* === Lifecycle === */
TieredMemory *tiered_mem_create(size_t l1_capacity, size_t l2_buckets, const char *l3_path);
void tiered_mem_free(TieredMemory *mem);

/* === L1 operations (fastest, direct pointer) === */
int tiered_mem_l1_put(TieredMemory *mem, uint64_t key, const float *data, size_t size);
const float *tiered_mem_l1_get(const TieredMemory *mem, uint64_t key, size_t *size_out);

/* === L2 operations (hashmap) === */
int tiered_mem_l2_put(TieredMemory *mem, uint64_t key, const float *data, size_t size);
const float *tiered_mem_l2_get(const TieredMemory *mem, uint64_t key, size_t *size_out);

/* === L3 operations (file-backed) === */
int tiered_mem_l3_put(TieredMemory *mem, uint64_t key, const float *data, size_t size);
float *tiered_mem_l3_get(const TieredMemory *mem, uint64_t key, size_t *size_out);

/* === Unified access with promotion === */
int tiered_mem_put(TieredMemory *mem, uint64_t key, const float *data, size_t size, MemTier tier);
float *tiered_mem_get(TieredMemory *mem, uint64_t key, size_t *size_out);

/* === Stats === */
void tiered_mem_stats(const TieredMemory *mem,
                      size_t *l1_count, size_t *l2_count,
                      size_t *hits, size_t *misses);

#ifdef __cplusplus
}
#endif

#endif /* PHYSMOL_MEMORY_H */
