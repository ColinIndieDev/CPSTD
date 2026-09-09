#include "../include/cpstd/hashmap.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void *hm_init_impl(size_t element_size, size_t capacity) {
    assert(capacity > 0);
    size_t final_capacity = capacity > 8 ? capacity : 8;
    hm_header_t *header = malloc(sizeof(hm_header_t) + (element_size * final_capacity));
    header->flag = malloc(sizeof(uint8_t) * final_capacity);
    memset(header->flag, HASH_EMPTY, sizeof(uint8_t) * final_capacity);
    header->size = 0;
    header->capacity = final_capacity;
    return (void *)(header + 1);
}

void hm_destroy(void *hm) {
    assert(hm);
    free(hm_header(hm));
}

void *hm_resize_impl(void *hm, size_t element_size, size_t key_size) {
    assert(hm);
    hm_header_t *header = hm_header(hm);
    size_t old_capacity = header->capacity;
    size_t new_capacity = old_capacity * 2;

    void *new_hm = hm_init_impl(element_size, new_capacity);
    hm_header_t *new_header = hm_header(new_hm);

    new_header->capacity = new_capacity;
    new_header->size = 0;

    for (size_t i = 0; i < old_capacity; i++) {
        if (header->flag[i] == HASH_OCCUPIED) {
            void *old_slot = (char *)hm + (i * element_size);
            size_t idx = hm_probe_impl(hm, old_slot, key_size, element_size);
            memcpy((char *)new_hm + (idx * element_size), old_slot, element_size);
            new_header->flag[idx] = HASH_OCCUPIED;
            new_header->size++;
        }
    }
    free(header->flag);
    free(header);

    return new_hm;
}

void *hm_begin(void *hm) {
    return hm;
}

void *hm_end_impl(void *hm, size_t element_size) {
    return hm + (hm_header(hm)->capacity * element_size);
}

int hm_it_exist_impl(void *hm, void *it, size_t element_size) {
    return hm_header(hm)->flag[(it - hm) / element_size] == HASH_OCCUPIED;
}
