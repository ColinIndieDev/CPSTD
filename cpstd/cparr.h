#pragma once

#define ARR_DEF(type, name)                                                    \
    typedef struct {                                                           \
        type *data;                                                            \
        u32 size;                                                              \
    } name;                                                                    \
    void name##_init(name *arr, u32 size) {                                    \
        arr->data = malloc(size * sizeof(type));                               \
        arr->size = size;                                                      \
    }                                                                          \
    type *name##_at(name *arr, u32 i) { return &arr->data[i]; }                \
    void name##_destroy(name *arr) {                                           \
        free(arr->data);                                                       \
        arr->size = 0;                                                         \
    }                                                                          \
    type *name##_begin(name *arr) { return arr->data; }                        \
    type *name##_end(name *arr) { return arr->data + arr->size; }

#define FOREACH_ARR(type, it, arrname, arrptr)                                 \
    for (type *it = arrname##_begin(arrptr); it != arrname##_end(arrptr); it++)
