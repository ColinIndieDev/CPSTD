#pragma once

#include <assert.h>

#include "cpbase.h"
#include "cpmemory.h"

typedef struct {
    u8 *data;
    u32 bits;
} bit_arr;

void bit_arr_init(bit_arr *a, u32 bits) {
    assert(a != NULLPTR);
    assert(bits > 0);
    a->bits = bits;
    u32 bytes = (bits + 7) / 8;
    a->data = cp_calloc(bytes, 1);
}
void bit_arr_destroy(bit_arr *a) {
    assert(a != NULLPTR);
    cp_free(a->data);
    a->bits = 0;
}
void bit_arr_set(bit_arr *a, u32 i) {
    assert(a != NULLPTR);
    assert(i >= 0 && i < a->bits);
    u32 byte = i >> 3;
    u32 bit = i & 7;
    a->data[byte] |= (1 << bit);
}
void bit_arr_clear(bit_arr *a, u32 i) {
    assert(a != NULLPTR);
    assert(i >= 0 && i < a->bits);
    u32 byte = i >> 3;
    u32 bit = i & 7;
    a->data[byte] &= ~(1 << bit);
}
b8 bit_arr_get(bit_arr *a, u32 i) {
    assert(a != NULLPTR);
    assert(i >= 0 && i < a->bits);
    u32 byte = i >> 3;
    u32 bit = i & 7;
    return (a->data[byte] >> bit) & 1;
}
