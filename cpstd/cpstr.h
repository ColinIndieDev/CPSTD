#pragma once

#include "cpbase.h"
#include <stdlib.h>

typedef struct {
    u8 *data;
    u32 size;
    b8 heap;
} str8;

#define STR8(t) (str8){t, sizeof(t) - 1, 0}
#define STR8_LIT(s) (s).data
#define STR8_RESERVE(l) (str8){malloc(l), l, 1}

#define STR8_LIT_CPY(s, t) do { (s).data = t; (s).size = sizeof(t) - 1; } while (0)
void str8_cpy(str8 *src, str8 *dest) {
    if (dest->heap && !src->heap) {
        free(dest->data);
    }
    dest->heap = src->heap;
    dest->data = src->data;
    dest->size = src->size;
}
b8 str8_cmp(str8 *s1, str8 *s2) {
    if (s1->size != s2->size) {
        return false;
    }
    const u8 *p1 = (const u8 *)s1->data;
    const u8 *p2 = (const u8 *)s2->data;
    while (*p1 && (*p1 == *p2)) {
        p1++;
        p2++;
    }
    return !(*p1 - *p2);
}
b8 str8_lit_cmp(str8 *s, const char *t) {
    const u8 *p1 = (const u8 *)s->data;
    const u8 *p2 = (const u8 *)t;
    while (*p1 && (*p1 == *p2)) {
        p1++;
        p2++;
    }
    return !(*p1 - *p2);
}
void str8_destroy(str8 *s) {
    if (s->heap) {
        free(s->data);
        s->heap = false;
    }
    s->size = 0;
}
u32 str8_len(str8 *s) {
    if (*s->data == '\0') {
        return 0;
    }
    u32 len = 0;
    while (s->data[len] != '\0') {
        len++;
    }
    return len;
}
