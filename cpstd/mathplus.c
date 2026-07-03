#include <mathplus.h>

float math_rad(float deg) {
    return deg * (float)(MATH_PI / 180.0);
}

int vec2f_cmp(vec2f a, vec2f b) { return a.x == b.x && a.y == b.y; }

vec2f vec2f_add(vec2f a, vec2f b) { return (vec2f){a.x + b.x, a.y + b.y}; }
vec2f vec2f_sub(vec2f a, vec2f b) { return (vec2f){a.x - b.x, a.y - b.y}; }
vec2f vec2f_mul(vec2f a, vec2f b) { return (vec2f){a.x * b.x, a.y * b.y}; }
vec2f vec2f_div(vec2f a, vec2f b) { return (vec2f){a.x / b.x, a.y / b.y}; }

vec2f vec2f_float_add(vec2f a, float b) { return (vec2f){a.x + b, a.y + b}; }
vec2f vec2f_float_sub(vec2f a, float b) { return (vec2f){a.x - b, a.y - b}; }
vec2f vec2f_float_mul(vec2f a, float b) { return (vec2f){a.x * b, a.y * b}; }
vec2f vec2f_float_div(vec2f a, float b) { return (vec2f){a.x / b, a.y / b}; }

float vec2f_dist(vec2f a, vec2f b) {
    float x = math_abs(a.x - b.x);
    float y = math_abs(a.y - b.y);

    return sqrtf((x * x) + (y * y));
}

float vec2f_dist2(vec2f a, vec2f b) {
    float x = math_abs(a.x - b.x);
    float y = math_abs(a.y - b.y);

    return (x * x) + (y * y);
}

float vec2f_dot(vec2f a, vec2f b) { return (a.x * b.x) + (a.y * b.y); }

float vec2f_length(vec2f a) { return sqrtf((a.x * a.x) + (a.y * a.y)); }

vec2f vec2f_clamp(vec2f v, vec2f n, vec2f m) {
    return (vec2f){math_clamp(v.x, n.x, m.x), math_clamp(v.y, n.y, m.y)};
}

vec2f vec2f_norm(vec2f v) {
    return (vec2f){v.x / sqrtf((v.x * v.x) + (v.y * v.y)),
                   v.y / sqrtf((v.x * v.x) + (v.y * v.y))};
}

void mat4f_identity(mat4f *m) {
    memset(*m, 0, sizeof(float) * 16);
    (*m)[0]  = 1.0f;
    (*m)[5]  = 1.0f;
    (*m)[10] = 1.0f;
    (*m)[15] = 1.0f;
}

void mat4f_translate(mat4f *m, vec3f v) {
    (*m)[12] += ((*m)[0] * v.x) + ((*m)[4] * v.y) + ((*m)[8] * v.z);
    (*m)[13] += ((*m)[1] * v.x) + ((*m)[5] * v.y) + ((*m)[9] * v.z);
    (*m)[14] += ((*m)[2] * v.x) + ((*m)[6] * v.y) + ((*m)[10] * v.z);
    (*m)[15] += ((*m)[3] * v.x) + ((*m)[7] * v.y) + ((*m)[11] * v.z);
}

void mat4f_scale(mat4f *m, vec3f v) {
    (*m)[0] *= v.x; 
    (*m)[1] *= v.x; 
    (*m)[2] *= v.x; 
    (*m)[3] *= v.x;
    (*m)[4] *= v.y; 
    (*m)[5] *= v.y; 
    (*m)[6] *= v.y; 
    (*m)[7] *= v.y;
    (*m)[8] *= v.z; 
    (*m)[9] *= v.z; 
    (*m)[10] *= v.z; 
    (*m)[11] *= v.z;
}

void mat4f_mul(mat4f *a, mat4f *b, mat4f *dest) {
    mat4f result;
    for (unsigned int c = 0; c < 4; c++) {
        for (unsigned int r = 0; r < 4; r++) {
            result[(c * 4) + r] =  ((*a)[(0 * 4) + r] * (*b)[(c * 4) + 0]) + ((*a)[(1 * 4) + r] * (*b)[(c * 4) + 1]) +
                                   ((*a)[(2 * 4) + r] * (*b)[(c * 4) + 2]) + ((*a)[(3 * 4) + r] * (*b)[(c * 4) + 3]);
        }
    }
    memcpy(*dest, result, sizeof(mat4f));
}

void mat4f_rotate(mat4f *m, float angle_rad, vec3f axis) {
    float c = cosf(angle_rad);
    float s = sinf(angle_rad);
    float t = 1.0f - c;

    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    mat4f rot;
    mat4f_identity(&rot);

    rot[0] = (t * x * x) + c;
    rot[1] = (t * x * y) + (s * z);
    rot[2] = (t * x * z) - (s * y);

    rot[4] = (t * x * y) - (s * z);
    rot[5] = (t * y * y) + c;
    rot[6] = (t * y * z) + (s * x);

    rot[8] = (t * x * z) + (s * y);
    rot[9] = (t * y * z) - (s * x);
    rot[10] = (t * z * z) + c;

    mat4f_mul(m, &rot, m);
}

vec4f mat4f_mul_vec4f(mat4f *m, vec4f v) {
    vec4f out;
    out.data[0] = ((*m)[0] * v.data[0]) + ((*m)[4] * v.data[1]) + ((*m)[8] * v.data[2]) + ((*m)[12] * v.data[3]);
    out.data[1] = ((*m)[1] * v.data[0]) + ((*m)[5] * v.data[1]) + ((*m)[9] * v.data[2]) + ((*m)[13] * v.data[3]);
    out.data[2] = ((*m)[2] * v.data[0]) + ((*m)[6] * v.data[1]) + ((*m)[10] * v.data[2]) + ((*m)[14] * v.data[3]);
    out.data[3] = ((*m)[3] * v.data[0]) + ((*m)[7] * v.data[1]) + ((*m)[11] * v.data[2]) + ((*m)[15] * v.data[3]);
    return out;
}

void mat4f_ortho(mat4f *m, float left, float right, float bottom, float top, float near, float far) {
    memset((*m), 0, sizeof(float) * 16);
    
    (*m)[0]  = 2.0f / (right - left);
    (*m)[5]  = 2.0f / (bottom - top);
    (*m)[10] = 1.0f / (far - near);   

    (*m)[12] = -(right + left) / (right - left);
    (*m)[13] = -(bottom + top) / (bottom - top);
    (*m)[14] = -near / (far - near);
    (*m)[15] = 1.0f;
}

static float minor_mat3f_det(const float *data, unsigned int r, unsigned int c) {
    float sub[3][3];
    unsigned int si = 0;
    for (unsigned int i = 0; i < 4; i++) {
        if (i == r) {
            continue;
        }
        unsigned int sj = 0;
        for (unsigned int j = 0; j < 4; j++) {
            if (j == c) {
                continue;
            }
            sub[si][sj++] = data[(j * 4) + i];
        }
        si++;
    }
    return (sub[0][0] * (sub[1][1] * sub[2][2] - sub[1][2] * sub[2][1])) - (sub[0][1] * (sub[1][0] * sub[2][2] - sub[1][2] * sub[2][0])) +
           (sub[0][2] * (sub[1][0] * sub[2][1] - sub[1][1] * sub[2][0]));
}

float mat4f_det(mat4f *m) {
    float det = 0.0f;
    for (unsigned int j = 0; j < 4; j++) {
        float cofactor = minor_mat3f_det((*m), 0, j);
        if (j % 2 != 0) {
            cofactor = -cofactor;
        }
        det += (*m)[(j * 4) + 0] * cofactor;
    }
    return det;
}

void mat4f_inv(mat4f *m, mat4f *out) {
    float cofactors[4][4];
    for (unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 4; j++) {
            float c = minor_mat3f_det((*m), i, j);
            if ((i + j) % 2 != 0) c = -c;
            cofactors[i][j] = c;
        }
    }
    
    float det = 0.0f;
    for (unsigned int j = 0; j < 4; j++) {
        det += (*m)[(j * 4) + 0] * cofactors[0][j];
    }
    
    float inv_det = 1.0f / det;
    for (unsigned int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 4; j++) {
            (*out)[(i * 4) + j] = cofactors[i][j] * inv_det;
        }
    }
}
