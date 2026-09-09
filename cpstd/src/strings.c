#include "../include/cpstd/strings.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

string *string_create(const char *cstr) {
    string *str = malloc(sizeof(string));
    if (cstr && strlen(cstr) > 0) {
        str->size = strlen(cstr);
        str->cstr = malloc(str->size);
        strncpy(str->cstr, cstr, str->size);
    } else {
        str->cstr = NULL;
        str->size = 0;
    }

    return str;
}

string *string_create_fmt(const char *fmt, ...) {
    string *str = malloc(sizeof(string));
    char buffer[1024];
    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, 1024, fmt, args);
        va_end(args);
    }

    if (fmt && strlen(buffer) > 0) {
        str->size = strlen(buffer);
        str->cstr = malloc(str->size);
        strncpy(str->cstr, buffer, str->size);
    } else {
        str->cstr = NULL;
        str->size = 0;
    }

    return str;
}

void string_destroy(string *str) {
    assert(str);
    if (str->cstr) {
        free(str->cstr);
    }
    free(str);
}

void string_cpy(string *dest, string *src) {
    assert(src);
    assert(dest);
    assert(src->cstr);
    if (dest->cstr) {
        free(dest->cstr);
    }
    dest->cstr = malloc(src->size);
    strncpy(dest->cstr, src->cstr, src->size);
    dest->size = src->size;
}

void string_cset(string *str, size_t i, char c) {
    assert(str);
    if (!str->cstr) {
        return;
    }
    assert(0 <= i && i < str->size);
    assert(c != '\0');
    str->cstr[i] = c;
}

char string_cget(string *str, size_t i) {
    assert(str);
    if (!str->cstr) {
        return '\0';
    }
    assert(0 <= i && i < str->size);
    return str->cstr[i];
}

int string_cmp(string *a, string *b) {
    assert(a);
    assert(b);
    if (!a && !b) {
        return 1;
    }
    if (a && b) {
        if (a->size == b->size) {
            for (size_t i = 0; i < a->size; i++) {
                if (a->cstr[i] != b->cstr[i]) {
                    return 0;
                }
            }
            return 1;
        }
        return 0;
    }
    return 0;
}

void string_view_create(string_view *view, const char *cstr) {
    (void)view;
    (void)cstr;
    assert(0 && "string_view_create() not implemented yet");
}
