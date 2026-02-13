#pragma once

#include "cpbase.h"
#include "cpvec.h"

#define PI 3.14159265358979323846f

VEC_DEF(f32, vecf)

typedef struct {
    vecf data;
    size_t rows, cols;
} mat2f;

void mat2f_init(mat2f *m, size_t rows, size_t cols, f32 val) {
    vecf_init(&m->data, rows * cols, val);
    m->rows = rows;
    m->cols = cols;
}

f32 *mat2f_at(mat2f *m, size_t row, size_t col) {
    assert(row < m->rows && col < m->cols);

    return &m->data.data[(row * m->cols) + col];
}
/*
[[nodiscard]] f32 get(const size_t row, const size_t col) const {
    if (row >= rows || col >= cols)
        return 0.0f;

    return data[(row * cols) + col];
}

f32 *rowPtr(const size_t r) { return data.data() + (r * cols); }
[[nodiscard]] std::vector<f32> getRow(const size_t r) const {
    std::vector<f32> row(cols);
    std::copy(data.begin() + scast(i32, r * cols),
              data.begin() + scast(i32, (r + 1) * cols), row.begin());
    return row;
}
[[nodiscard]] const f32 *rowPtr(const size_t r) const {
    return data.data() + (r * cols);
}

[[nodiscard]] size_t size() const { return rows * cols; }
*/

/*
struct Vec3f {
    f32 x, y, z;

    Vec3f(const f32 x, const f32 y, const f32 z) : x(x), y(y), z(z) {}
    Vec3f(const f32 val) : x(val), y(val), z(val) {}
};

struct Vec3i {
    i32 x, y, z;

    Vec3i(const i32 x, const i32 y, const i32 z) : x(x), y(y), z(z) {}
    Vec3i(const i32 val) : x(val), y(val), z(val) {}
};

Mat2f AddMat2f(const Mat2f &a, const Mat2f &b);
Mat2f SubMat2f(const Mat2f &a, const Mat2f &b);

void PrintMat2f(Mat2f &m);
void PrintMat2f(Mat2f m);
*/

/*

i32 max(i32 a, i32 b);
f32 maxf(f32 a, f32 b);
i32 min(i32 a, i32 b);
f32 minf(f32 a, f32 b);

i32 abs(i32 x);
f32 absf(f32 x);

i32 clamp(i32 x, i32 n, i32 m);
f32 clampf(f32 x, f32 n, f32 m);

f32 factorial(i32 n);
f32 expf(f32 x, i32 termsCount = 20);

f32 powf(f32 x, i32 n);

f32 sinf(f32 x, i32 n = 10);
f32 cosf(f32 x, i32 n = 10);
f32 tanf(f32 x);
f32 sinhf(f32 x);
f32 coshf(f32 x);
f32 tanhf(f32 x);

*/

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
