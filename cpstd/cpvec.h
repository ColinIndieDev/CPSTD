#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VEC_DEF(type, name)                                                    \
    typedef struct {                                                           \
        type *data;                                                            \
        size_t size;                                                           \
        size_t capacity;                                                       \
    } name;                                                                    \
    void name##_init(name *vec, size_t size, type val) {                       \
        vec->capacity = size > 10 ? size * 2 : 10;                             \
        vec->data = malloc(vec->capacity * sizeof(type));                      \
        vec->size = size;                                                      \
        for (size_t i = 0; i < vec->size; i++) {                               \
            vec->data[i] = val;                                                \
        }                                                                      \
    }                                                                          \
    void name##_reserve(name *vec, size_t size) {                              \
        vec->capacity = size;                                                  \
        vec->data = malloc(vec->capacity * sizeof(type));                      \
        vec->size = 0;                                                         \
    }                                                                          \
    type *name##_at(name *vec, size_t i) {                                     \
        assert("Out of bounds or not initialized" && 0 <= i &&                 \
               i < vec->size && vec->size > 0);                                \
        return &vec->data[i];                                                  \
    }                                                                          \
    type name##_get(name *vec, size_t i) {                                     \
        assert("Out of bounds or not initialized" && 0 <= i &&                 \
               i < vec->size && vec->size > 0);                                \
        return vec->data[i];                                                   \
    }                                                                          \
    void name##_push_back(name *vec, type val) {                               \
        if (vec->size >= vec->capacity) {                                      \
            vec->capacity *= 2;                                                \
            vec->data = realloc(vec->data, vec->capacity * sizeof(type));      \
        }                                                                      \
        vec->data[vec->size] = val;                                            \
        vec->size++;                                                           \
    }                                                                          \
    void name##_push_front(name *vec, type val) {                              \
        if (vec->size >= vec->capacity) {                                      \
            vec->capacity *= 2;                                                \
            vec->data = realloc(vec->data, vec->capacity * sizeof(type));      \
        }                                                                      \
        memmove(&vec->data[1], &vec->data[0], vec->size * sizeof(type));       \
        vec->data[0] = val;                                                    \
        vec->size++;                                                           \
    }                                                                          \
    void name##_pop_back(name *vec) {                                          \
        assert("Cannot pop element if there are none" && vec->size > 0);       \
        vec->size--;                                                           \
    }                                                                          \
    void name##_pop_front(name *vec) {                                         \
        assert("Cannot pop element if there are none" && vec->size > 0);       \
        memmove(&vec->data[0], &vec->data[1], (vec->size - 1) * sizeof(type)); \
        vec->size--;                                                           \
    }                                                                          \
    void name##_delete(name *vec, size_t i) {                                  \
        assert("Out of bounds or not initialized" && 0 <= i &&                 \
               i < vec->size && vec->size > 0);                                \
        memmove(&vec->data[i], &vec->data[i + 1],                              \
                (vec->size - i - 1) * sizeof(type));                           \
        vec->size--;                                                           \
    }                                                                          \
    void name##_destroy(name *vec) {                                           \
        free(vec->data);                                                       \
        vec->size = 0;                                                         \
        vec->capacity = 0;                                                     \
    }                                                                          \
    void name##_set(name *vec, size_t i, type val) {                           \
        assert("Out of bounds or not initialized" && 0 <= i &&                 \
               i < vec->size && vec->size > 0);                                \
        vec->data[i] = val;                                                    \
    }
