#pragma once

#include <sys/mman.h>
#include <unistd.h>

#include "cpbase.h"

typedef struct block_meta_ {
    u32 size;
    struct block_meta_ *next;
    b8 is_mmap;
    b8 free;
} block_meta;

#define META_SIZE sizeof(block_meta)
#define MMAP_THRESHOLD (128 * 1024)

block_meta *free_list = NULLPTR;

void coalesce_block(block_meta *block);
block_meta *request_space(block_meta *last, u32 size);
block_meta *find_free_block(block_meta **last, u32 size);
void split_block(block_meta *block, u32 size);

void *cp_memset(void *ptr, i32 val, u32 size) {
    u8 *p = (u8 *)ptr;
    while (size--) {
        *p++ = (u8)val;
    }
    return ptr;
}

void *cp_memcpy(void *dest, const void *src, u32 size) {
    u8 *d = (u8 *)dest;
    const u8 *s = (const u8 *)src;
    while (size--) {
        *d++ = *s++;
    }
    return dest;
}

void *cp_memmove(void *dest, const void *src, u32 size) {
    u8 *d = (u8 *)dest;
    const u8 *s = (const u8 *)src;

    if (d == s) {
        return dest;
    }

    if (d < s) {
        while (size--) {
            *d++ = *s++;
        }
    } else {
        d += size;
        s += size;
        while (size--) {
            *--d = *--s;
        }
    }
    return dest;
}

int cp_memcmp(const void *ptr1, const void *ptr2, u32 size) {
    const u8 *a = (const u8 *)ptr1;
    const u8 *b = (const u8 *)ptr2;
    while (size--) {
        if (*a != *b) {
            return *a - *b;
        }
        a++;
        b++;
    }
    return 0;
}

void *cp_malloc(u32 size) {
    block_meta *block;
    if (size == 0) {
        return NULLPTR;
    }
    size = (size + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
    if (size >= MMAP_THRESHOLD) {
        void *ptr = mmap(NULLPTR, size + META_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return NULLPTR;
        }
        block = (block_meta *)ptr;
        block->size = size;
        block->next = NULLPTR;
        block->free = false;
        block->is_mmap = true;
    } else {
        if (!free_list) {
            block = request_space(NULLPTR, size);
            if (!block) {
                return NULLPTR;
            }
            free_list = block;
        } else {
            block_meta *last = free_list;
            block = find_free_block(&last, size);
            if (!block) {
                block = request_space(last, size);
                if (!block) {
                    return NULLPTR;
                }
            } else {
                split_block(block, size);
                block->free = false;
                block->is_mmap = false;
            }
        }
    }
    return (block + 1);
}

void cp_free(void *ptr) {
    if (!ptr) {
        return;
    }
    block_meta *block = (block_meta *)ptr - 1;
    if (block->is_mmap) {
        munmap(block, block->size + META_SIZE);
    } else {
        block->free = true;
        coalesce_block(block);
    }
}

void *cp_calloc(u32 num, u32 size) {
    if (num != 0 && size > U32_MAX / num) {
        return NULLPTR;
    }
    u32 total = num * size;
    void *ptr = cp_malloc(total);
    if (!ptr) {
        return NULLPTR;
    }
    cp_memset(ptr, 0, total);
    return ptr;
}

void *cp_realloc(void *ptr, u32 size) {
    if (!ptr) {
        return cp_malloc(size);
    }
    if (size == 0) {
        cp_free(ptr);
        return NULLPTR;
    }
    block_meta *block = (block_meta *)ptr - 1;
    if (block->size >= size) {
        return ptr;
    }
    void *new_ptr = cp_malloc(size);
    if (!new_ptr) {
        return NULLPTR;
    }
    cp_memcpy(new_ptr, ptr, block->size);
    cp_free(ptr);

    return new_ptr;
}

void coalesce_block(block_meta *block) {
    block_meta *prev = NULLPTR;
    block_meta *current = free_list;
    while (current && current->next != block) {
        current = current->next;
    }
    prev = current;
    if (block->next && block->next->free) {
        block->size += META_SIZE + block->next->size;
        block->next = block->next->next;
    }
    if (prev && prev->free) {
        prev->size += META_SIZE + block->size;
        prev->next = block->next;
    }
}

block_meta *request_space(block_meta *last, u32 size) {
    void *request = sbrk((u32)(size + META_SIZE));
    if (request == (void *)-1) {
        return NULLPTR;
    }
    block_meta *block = (block_meta *)request;
    block->size = size;
    block->next = NULLPTR;
    block->free = false;
    block->is_mmap = false;
    if (last) {
        last->next = block;
    }
    return block;
}

block_meta *find_free_block(block_meta **last, u32 size) {
    block_meta *cur = free_list;
    while (cur && !(cur->free && cur->size >= size)) {
        if (cur != free_list) {
            *last = cur;
        }
        cur = cur->next;
    }
    return cur;
}

void split_block(block_meta *block, u32 size) {
    if (block->size >= size + META_SIZE + sizeof(void *)) {
        block_meta *new_block = (block_meta *)((char *)(block + 1) + size);
        new_block->size = block->size - size - META_SIZE;
        new_block->next = block->next;
        new_block->free = true;
        new_block->is_mmap = false;
        block->size = size;
        block->next = new_block;
    }
}
