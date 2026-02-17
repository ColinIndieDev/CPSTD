#pragma once

#include <stdlib.h>

#include "cpbase.h"

typedef struct {
    u8 *data;
    u64 size;
    u64 capacity;
} str8;

#define LIT_TO_STR8(s) (str8){(u8 *)(s), sizeof((s)) - 1, sizeof((s)) - 1}
#define STR8(str, size)                                                        \
    (str8) { (u8 *)(str), size }
#define STR8_FMT_OUT(s8) (int)(s8).size, (s8).data

str8 *init_str8(u64 capacity) {
    str8 *str;
    str->data = malloc(capacity);
    str->size = 0;
    str->capacity = capacity;
    return str;
}
