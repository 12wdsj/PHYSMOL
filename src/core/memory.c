#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* FNV-1a hash */
static uint64_t fnv1a_hash(uint64_t key) {
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < 8; i++) {
        hash ^= (key >> (i * 8)) & 0xFF;
        hash *= 1099511628211ULL;
    }
    return hash;
}

/* === L2 HashMap === */

static MemHashMap hashmap_create(size_t num_buckets) {
    MemHashMap hm;
    hm.num_buckets = num_buckets;
    hm.num_entries = 0;
    hm.buckets = (MemEntry **)calloc(num_buckets, sizeof(MemEntry *));
    return hm;
}

static void hashmap_free(MemHashMap *hm) {
    for (size_t i = 0; i < hm->num_buckets; i++) {
        MemEntry *e = hm->buckets[i];
        while (e) {
            MemEntry *next = e->next;
            free(e->data);
            free(e);
            e = next;
        }
    }
    free(hm->buckets);
    hm->buckets = NULL;
    hm->num_entries = 0;
}

static int hashmap_put(MemHashMap *hm, uint64_t key, const float *data, size_t size) {
    uint64_t h = fnv1a_hash(key) % hm->num_buckets;

    /* Check if key exists */
    for (MemEntry *e = hm->buckets[h]; e; e = e->next) {
        if (e->key == key) {
            /* Update */
            float *new_data = (float *)realloc(e->data, size * sizeof(float));
            if (!new_data) return -1;
            e->data = new_data;
            memcpy(e->data, data, size * sizeof(float));
            e->size = size;
            return 0;
        }
    }

    /* New entry */
    MemEntry *e = (MemEntry *)malloc(sizeof(MemEntry));
    if (!e) return -1;
    e->key = key;
    e->size = size;
    e->data = (float *)malloc(size * sizeof(float));
    if (!e->data) { free(e); return -1; }
    memcpy(e->data, data, size * sizeof(float));
    e->next = hm->buckets[h];
    hm->buckets[h] = e;
    hm->num_entries++;
    return 0;
}

static const float *hashmap_get(const MemHashMap *hm, uint64_t key, size_t *size_out) {
    uint64_t h = fnv1a_hash(key) % hm->num_buckets;
    for (MemEntry *e = hm->buckets[h]; e; e = e->next) {
        if (e->key == key) {
            if (size_out) *size_out = e->size;
            return e->data;
        }
    }
    return NULL;
}

/* === TieredMemory === */

TieredMemory *tiered_mem_create(size_t l1_capacity, size_t l2_buckets, const char *l3_path) {
    TieredMemory *mem = (TieredMemory *)calloc(1, sizeof(TieredMemory));
    if (!mem) return NULL;

    mem->l1_capacity = l1_capacity ? l1_capacity : 1024;
    mem->l1_slots = (float **)calloc(mem->l1_capacity, sizeof(float *));
    mem->l1_sizes = (size_t *)calloc(mem->l1_capacity, sizeof(size_t));
    mem->l1_count = 0;

    mem->l2 = hashmap_create(l2_buckets ? l2_buckets : 4096);

    mem->l3_path = l3_path ? strdup(l3_path) : NULL;

    return mem;
}

void tiered_mem_free(TieredMemory *mem) {
    if (!mem) return;
    for (size_t i = 0; i < mem->l1_count; i++) {
        free(mem->l1_slots[i]);
    }
    free(mem->l1_slots);
    free(mem->l1_sizes);
    hashmap_free(&mem->l2);
    free(mem->l3_path);
    free(mem);
}

/* L1: store in next available slot */
int tiered_mem_l1_put(TieredMemory *mem, uint64_t key, const float *data, size_t size) {
    /* For L1, we use key as slot index (simple direct-mapped) */
    size_t slot = key % mem->l1_capacity;

    /* Free existing */
    free(mem->l1_slots[slot]);

    mem->l1_slots[slot] = (float *)malloc(size * sizeof(float));
    if (!mem->l1_slots[slot]) return -1;
    memcpy(mem->l1_slots[slot], data, size * sizeof(float));
    mem->l1_sizes[slot] = size;

    if (slot >= mem->l1_count) mem->l1_count = slot + 1;
    return 0;
}

const float *tiered_mem_l1_get(const TieredMemory *mem, uint64_t key, size_t *size_out) {
    size_t slot = key % mem->l1_capacity;
    if (slot < mem->l1_capacity && mem->l1_slots[slot]) {
        if (size_out) *size_out = mem->l1_sizes[slot];
        return mem->l1_slots[slot];
    }
    return NULL;
}

int tiered_mem_l2_put(TieredMemory *mem, uint64_t key, const float *data, size_t size) {
    return hashmap_put(&mem->l2, key, data, size);
}

const float *tiered_mem_l2_get(const TieredMemory *mem, uint64_t key, size_t *size_out) {
    return hashmap_get(&mem->l2, key, size_out);
}

int tiered_mem_l3_put(TieredMemory *mem, uint64_t key, const float *data, size_t size) {
    if (!mem->l3_path) return -1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/%lu.dat", mem->l3_path, (unsigned long)key);

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    fwrite(&size, sizeof(size_t), 1, f);
    fwrite(data, sizeof(float), size, f);
    fclose(f);
    return 0;
}

float *tiered_mem_l3_get(const TieredMemory *mem, uint64_t key, size_t *size_out) {
    if (!mem->l3_path) return NULL;

    char path[1024];
    snprintf(path, sizeof(path), "%s/%lu.dat", mem->l3_path, (unsigned long)key);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    size_t size;
    fread(&size, sizeof(size_t), 1, f);

    float *data = (float *)malloc(size * sizeof(float));
    if (!data) { fclose(f); return NULL; }

    fread(data, sizeof(float), size, f);
    fclose(f);

    if (size_out) *size_out = size;
    return data;
}

/* Unified access: try L1 -> L2 -> L3, promote on hit */
float *tiered_mem_get(TieredMemory *mem, uint64_t key, size_t *size_out) {
    /* Try L1 */
    const float *l1 = tiered_mem_l1_get(mem, key, size_out);
    if (l1) {
        mem->l1_hits++;
        return (float *)l1; /* const cast ok for read-only use */
    }

    /* Try L2 */
    const float *l2 = tiered_mem_l2_get(mem, key, size_out);
    if (l2) {
        mem->l2_hits++;
        /* Promote to L1 */
        tiered_mem_l1_put(mem, key, l2, *size_out);
        return (float *)l2;
    }

    /* Try L3 */
    float *l3 = tiered_mem_l3_get(mem, key, size_out);
    if (l3) {
        mem->l3_hits++;
        /* Promote to L1 and L2 */
        tiered_mem_l1_put(mem, key, l3, *size_out);
        tiered_mem_l2_put(mem, key, l3, *size_out);
        return l3;
    }

    mem->misses++;
    return NULL;
}

int tiered_mem_put(TieredMemory *mem, uint64_t key, const float *data, size_t size, MemTier tier) {
    switch (tier) {
        case MEM_TIER_L1_HOT:  return tiered_mem_l1_put(mem, key, data, size);
        case MEM_TIER_L2_WARM: return tiered_mem_l2_put(mem, key, data, size);
        case MEM_TIER_L3_COLD: return tiered_mem_l3_put(mem, key, data, size);
        default: return -1;
    }
}

void tiered_mem_stats(const TieredMemory *mem,
                      size_t *l1_count, size_t *l2_count,
                      size_t *hits, size_t *misses) {
    if (l1_count) *l1_count = mem->l1_count;
    if (l2_count) *l2_count = mem->l2.num_entries;
    if (hits) *hits = mem->l1_hits + mem->l2_hits + mem->l3_hits;
    if (misses) *misses = mem->misses;
}
