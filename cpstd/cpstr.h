#pragma once

#include "cpbase.h"

typedef struct {
    u8 *data;
    u64 size;
} str8;

#define TO_STR8(s) (str8){(u8 *)(s), sizeof((s)) - 1}
#define STR8_FMT_OUT(s8) (int)(s8).size, (s8).data
