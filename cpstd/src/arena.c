#include "../include/cpstd/arena.h"

mem_arena *arena_create(size_t capacity) {
    assert(capacity > 0);
    mem_arena *arena = malloc(capacity);
    arena->capacity = capacity;
    arena->pos = ARENA_BASE_POS;

    return arena;
}

void arena_destroy(mem_arena *arena) { free(arena); }

void *arena_alloc(mem_arena *arena, size_t size) {
    size_t pos_aligned = ALIGN_POW2(arena->pos, ARENA_ALIGN);
    size_t new_pos = pos_aligned + size;
    if (new_pos > arena->capacity) {
        return NULL;
    }
    arena->pos = new_pos;
    uint8_t *out = (uint8_t *)arena + pos_aligned;

    return out;
}

void *arena_calloc(mem_arena *arena, size_t size) {
    size_t pos_aligned = ALIGN_POW2(arena->pos, ARENA_ALIGN);
    size_t new_pos = pos_aligned + size;
    if (new_pos > arena->capacity) {
        return NULL;
    }
    arena->pos = new_pos;
    uint8_t *out = (uint8_t *)arena + pos_aligned;
    memset(out, 0, size);

    return out;
}

void arena_free(mem_arena *arena, size_t size) {
    size = (size < arena->pos - ARENA_BASE_POS) ? size
                                                : arena->pos - ARENA_BASE_POS;
    arena->pos -= size;
}

void arena_free_to(mem_arena *arena, size_t pos) {
    size_t size = pos < arena->pos ? arena->pos - pos : 0;
    arena_free(arena, size);
}

void arena_clear(mem_arena *arena) { arena_free_to(arena, ARENA_BASE_POS); }
