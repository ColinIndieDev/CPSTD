#include "../include/cpstd/hashset.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void *hs_init_impl(size_t element_size, size_t capacity) {
    assert(capacity > 0);
    size_t final_capacity = capacity > 8 ? capacity : 8;
    hs_header_t *header = malloc(sizeof(hs_header_t) + (element_size * final_capacity));
    assert(header);
    header->flag = malloc(sizeof(uint8_t) * final_capacity);
    assert(header->flag);
    memset(header->flag, HS_EMPTY, sizeof(uint8_t) * final_capacity);
    header->size = 0;
    header->capacity = final_capacity;
    return (void *)(header + 1);
}

void hs_destroy(void *hs) {
    assert(hs);
    hs_header_t *header = hs_header(hs);
    free(header->flag);
    free(header);
}

void *hs_resize_impl(void *hs, size_t element_size, size_t key_size) {
    assert(hs);
    hs_header_t *header = hs_header(hs);
    size_t old_capacity = header->capacity;
    size_t new_capacity = old_capacity * 2;
    void *new_hs = hs_init_impl(element_size, new_capacity);
    hs_header_t *new_header = hs_header(new_hs);
    for (size_t i = 0; i < old_capacity; i++) {
        if (header->flag[i] != HS_OCCUPIED) {
            continue;
        }
        void *old_slot = (char *)hs + (i * element_size);
        size_t idx = hs_probe_impl(new_hs, old_slot, key_size, element_size);
        memcpy((char *)new_hs + (idx * element_size), old_slot, element_size);
        new_header->flag[idx] = HS_OCCUPIED;
        new_header->size++;
    }
    free(header->flag);
    free(header);
    return new_hs;
}

void *hs_begin(void *hs) {
    return hs;
}

void *hs_end_impl(void *hs, size_t element_size) {
    return (char *)hs + (hs_header(hs)->capacity * element_size);
}

int hs_it_exist_impl(void *hs, void *it, size_t element_size) {
    hs_header_t *header = hs_header(hs);
    size_t index = ((char *)it - (char *)hs) / element_size;
    return header->flag[index] == HS_OCCUPIED;
}
