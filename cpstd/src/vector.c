#include "../include/cpstd/vector.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

void *vec_init_impl(size_t element_size, size_t capacity) {
    assert(capacity > 0);
    vec_header_t *header =
        malloc(sizeof(vec_header_t) + (element_size * capacity));
    header->size = 0;
    header->capacity = capacity;
    return (void *)(header + 1);
}

void vec_destroy(void *arr) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    free(header);
}

void vec_clear(void *arr) {
    assert(arr);
    vec_header(arr)->size = 0;
}

void *vec_push_impl(void *arr, void *val, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    if (header->size >= header->capacity) {
        size_t new_capacity = header->capacity * 2;
        vec_header_t *tmp = realloc(header, sizeof(vec_header_t) +
                                                (element_size * new_capacity));
        assert(tmp);
        header = tmp;
        header->capacity = new_capacity;
    }
    arr = (void *)(header + 1);
    memcpy((char *)arr + (header->size * element_size), val, element_size);
    header->size++;
    return arr;
}

void *vec_push_front_impl(void *arr, void *val, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    if (header->size >= header->capacity) {
        size_t new_capacity = header->capacity * 2;
        vec_header_t *tmp = realloc(header, sizeof(vec_header_t) +
                                                (element_size * new_capacity));
        assert(tmp);
        header = tmp;
        header->capacity = new_capacity;
    }
    arr = (void *)(header + 1);
    memmove(arr + element_size, arr, (header->size * element_size));
    memcpy(arr, val, element_size);
    header->size++;
    return arr;
}

void vec_pop(void *arr) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    assert(header->size > 0);
    header->size--;
}

void vec_pop_front_impl(void *arr, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    assert(header->size > 0);
    memmove(arr, arr + element_size, (header->size - 1) * element_size);
    header->size--;
}

void vec_delete_impl(void *arr, size_t i, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    assert(0 <= i && i < header->size && header->size > 0);
    memmove(arr + (i * element_size), arr + ((i + 1) * element_size),
            (header->size - i - 1) * element_size);
    header->size--;
}

void *vec_front(void *arr) {
    assert(arr);
    return arr;
}

void *vec_back_impl(void *arr, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    return arr + ((header->size - 1) * element_size);
}

void *vec_begin(void *arr) {
    assert(arr);
    return arr;
}

void *vec_end_impl(void *arr, size_t element_size) {
    assert(arr);
    vec_header_t *header = vec_header(arr);
    return arr + (header->size * element_size);
}
