#pragma once

#include "cpbase.h"
#include "cpvec.h"
#include <string.h>

#define PI 3.14159265358979323846f

#define CP_MIN(x, y) ((x) < (y) ? (x) : (y))
#define CP_MAX(x, y) ((x) > (y) ? (x) : (y))
#define CP_ABS(x) ((x) > 0 ? (x) : -(x))
#define CP_CLAMP(x, n, m) ((x) > (n) ? ((x) < (m) ? (x) : (m)) : (n))

VEC_DEF(f32, vecf)

typedef struct {
    vecf data;
    u32 rows, cols;
} mat2f;

void mat2f_init(mat2f *m, u32 rows, u32 cols, f32 val) {
    vecf_init(&m->data, rows * cols, val);
    m->rows = rows;
    m->cols = cols;
}

f32 *mat2f_at(mat2f *m, u32 row, u32 col) {
    assert(row < m->rows && col < m->cols);

    return &m->data.data[(row * m->cols) + col];
}

f32 mat2f_get(mat2f *m, u32 row, u32 col) {
    assert(row < m->rows && col < m->cols);

    return m->data.data[(row * m->cols) + col];
}

f32 *mat2f_row_ptr(mat2f *m, u32 row) { return &m->data.data[row * m->cols]; }
void mat2f_get_row(mat2f *m, u32 row, f32 *out) {
    assert(r < m->rows);

    memcpy(out, &m->data.data[row * m->cols], m->cols * sizeof(f32));
}

u32 mat2f_size(mat2f *m) { return m->data.size; }

void mat2f_add(mat2f *m1, mat2f *m2, mat2f *out) {
    assert(m1->rows == m2->rows && m1->cols == m2->cols &&
           out->rows == m1->rows && out->cols == m1->cols);

    for (u32 i = 0; i < m1->data.size; i++) {
        out->data.data[i] = m1->data.data[i] + m2->data.data[i];
    }
}
void mat2f_sub(mat2f *m1, mat2f *m2, mat2f *out) {
    assert(m1->rows == m2->rows && m1->cols == m2->cols &&
           out->rows == m1->rows && out->cols == m1->cols);

    for (u32 i = 0; i < m1->data.size; i++) {
        out->data.data[i] = m1->data.data[i] - m2->data.data[i];
    }
}
void mat2f_mul(mat2f *m1, mat2f *m2, mat2f *out) {
    assert(m1->rows == m2->rows && m1->cols == m2->cols &&
           out->rows == m1->rows && out->cols == m1->cols);

    for (u32 i = 0; i < m1->data.size; i++) {
        out->data.data[i] = m1->data.data[i] * m2->data.data[i];
    }
}
void mat2f_div(mat2f *m1, mat2f *m2, mat2f *out) {
    assert(m1->rows == m2->rows && m1->cols == m2->cols &&
           out->rows == m1->rows && out->cols == m1->cols);

    for (u32 i = 0; i < m1->data.size; i++) {
        out->data.data[i] = m1->data.data[i] / m2->data.data[i];
    }
}

typedef struct {
    f32 x, y, z;
} vec3f;
typedef struct {
    f32 x, y;
} vec2f;

f32 cp_factorial(i32 n) {
    if (n == 0 || n == 1) {
        return 1.0f;
    }
    f32 result = 1.0f;
    for (int i = 2; i <= n; i++) {
        result *= (f32)i;
    }
    return result;
}

f32 cp_expf(f32 x) {
    f32 result = 1.0f;
    f32 term = 1.0f;

    i32 termsCount = 20;
    for (int n = 1; n <= termsCount; n++) {
        term *= x / (f32)n;
        result += term;
    }
    return result;
}

f32 cp_powf(f32 x, i32 n) {
    f32 result = 1.0f;
    for (int i = 0; i < n; i++) {
        result *= x;
    }
    return n > 0 ? result : 1 / result;
}

f32 cp_sinf(f32 x) {
    f32 result = 0.0f;

    while (x > PI) {
        x -= 2 * PI;
    }
    while (x < -PI) {
        x += 2 * PI;
    }

    i32 n = 10;
    for (int i = 0; i < n; i++) {
        f32 sign = cp_powf(-1, i);
        f32 num = cp_powf(x, (2 * i) + 1);
        f32 den = cp_factorial((2 * i) + 1);
        result += sign * (num / den);
    }
    return result;
}

f32 cp_cosf(f32 x) {
    f32 result = 1.0f;
    f32 term = 1.0f;

    while (x > PI) {
        x -= 2 * PI;
    }
    while (x < -PI) {
        x += 2 * PI;
    }

    i32 n = 10;
    for (int i = 1; i <= n; i++) {
        f32 sign = -cp_powf(x, 2);
        f32 num = 2.0f * (f32)i;
        term *= sign / (num * (num - 1));
        result += term;
    }
    return result;
}

f32 cp_tanf(f32 x) { return cp_sinf(x) / cp_cosf(x); }

f32 cp_sinhf(f32 x) { return (cp_expf(x) - cp_expf(-x)) / 2; }

f32 cp_coshf(f32 x) { return (cp_expf(x) + cp_expf(-x)) / 2; }

f32 cp_tanhf(f32 x) { return cp_sinhf(x) / cp_coshf(x); }

f32 cp_sqrt(f32 n) {
    if (n < 0) {
        return -1.0f;
    }
    if (n == 0) {
        return 0.0f;
    }
    f32 tolerance = 1e-5f;
    f32 guess = n / 2.0f;
    while (true) {
        f32 newGuess = (guess + (n / guess)) / 2.0f;
        if (CP_ABS((newGuess * newGuess) - n) < tolerance) {
            return newGuess;
        }
        guess = newGuess;
    }
}

vec2f vec2f_add(vec2f *a, vec2f *b) {
    return (vec2f){a->x + b->x, a->y + b->y};
}
vec2f vec2f_sub(vec2f *a, vec2f *b) {
    return (vec2f){a->x - b->x, a->y - b->y};
}
vec2f vec2f_mul(vec2f *a, vec2f *b) {
    return (vec2f){a->x * b->x, a->y * b->y};
}
vec2f vec2f_div(vec2f *a, vec2f *b) {
    return (vec2f){a->x / b->x, a->y / b->y};
}

f32 vec2f_dist(vec2f *v1, vec2f *v2) {
    f32 a = CP_ABS(v1->x - v2->x);
    f32 b = CP_ABS(v1->y - v2->y);

    return cp_sqrt((a * a) + (b * b));
}

f32 vec2f_dist2(vec2f *v1, vec2f *v2) {
    f32 a = CP_ABS(v1->x - v2->x);
    f32 b = CP_ABS(v1->y - v2->y);

    return (a * a) + (b * b);
}

f32 vec2f_dot(vec2f *a, vec2f *b) { return (a->x * b->x) + (a->y * b->y); }

void mat2f_print(mat2f *m) {
    for (int r = 0; r < m->rows; r++) {
        for (int c = 0; c < m->cols; c++) {
            printf("%f", *mat2f_at(m, r, c));

            if (c < m->cols - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
}
