#pragma once

#include <assert.h>

#include "cpbase.h"
#include "cpmemory.h"

#define ARR_DEF(type, name)                                                    \
    typedef struct {                                                           \
        type *data;                                                            \
        u32 size;                                                              \
    } name;                                                                    \
    void name##_init(name *a, u32 size) {                                      \
        assert(a != NULLPTR);                                                  \
        assert(size > 0);                                                      \
        a->data = cp_malloc(size * sizeof(type));                              \
        a->size = size;                                                        \
    }                                                                          \
    type *name##_at(name *a, u32 i) {                                          \
        assert(a != NULLPTR);                                                  \
        assert(i >= 0 && i < a->size);                                         \
        return &a->data[i];                                                    \
    }                                                                          \
    void name##_destroy(name *a) {                                             \
        assert(a != NULLPTR);                                                  \
        cp_free(a->data);                                                      \
        a->size = 0;                                                           \
    }                                                                          \
    type *name##_begin(name *a) {                                              \
        assert(a != NULLPTR);                                                  \
        return a->data;                                                        \
    }                                                                          \
    type *name##_end(name *a) {                                                \
        assert(a != NULLPTR);                                                  \
        return a->data + a->size;                                              \
    }                                                                          \
    type *name##_front(name *a) {                                              \
        assert(a != NULLPTR);                                                  \
        return &a->data[0];                                                    \
    }                                                                          \
    type *name##_back(name *a) {                                               \
        assert(a != NULLPTR);                                                  \
        return &a->data[a->size - 1];                                          \
    }                                                                          \
    b8 name##_empty(name *a) {                                                 \
        assert(a != NULLPTR);                                                  \
        return a->size <= 0;                                                   \
    }

#define FOREACH_ARR(type, aname, it, aptr)                                     \
    for (type *it = aname##_begin(aptr); it != aname##_end(aptr); it++)
