#include "queue.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void *queue_init_impl(size_t capacity, size_t element_size) {
    assert(capacity > 0);
    q_header_t *header = malloc(sizeof(q_header_t) + (capacity * element_size));
    header->size = 0;
    header->capacity = capacity;
    header->head = 0;
    header->tail = 0;

    return (void *)(header + 1);
}

void queue_destroy(void *q) {
    assert(q);
    free(q_header(q));
}

void *queue_resize_impl(void *q, size_t element_size) {
    assert(q);
    q_header_t *header = q_header(q);
    size_t new_capacity = header->capacity * 2;
    void *new_queue = queue_init_impl(new_capacity, element_size);
    for (size_t i = 0; i < header->size; i++) {
        memcpy(new_queue + (i * element_size),
               q + (((header->tail + i) % header->capacity) * element_size),
               element_size);
    } 
    q_header_t *new_header = q_header(new_queue);
    new_header->capacity = new_capacity;
    new_header->tail = 0;
    new_header->head = header->size; 
    new_header->size = header->size;

    free(header);

    return new_queue;
}
