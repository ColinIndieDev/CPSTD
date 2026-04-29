#pragma once

#define _GNU_SOURCE

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <malloc.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef CPL_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define MINIAUDIO_IMPLEMENTATION
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

#include "stb_image.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include "stb_image_write.h"
#include <miniaudio.h>

#include "../cpstd/cpbase.h"
#include "../cpstd/cpmath.h"
#include "../cpstd/cpvec.h"

// {{{ Key Inputs

#define MOUSE_BUTTON_1 0
#define MOUSE_BUTTON_2 1
#define MOUSE_BUTTON_3 2
#define MOUSE_BUTTON_4 3
#define MOUSE_BUTTON_5 4
#define MOUSE_BUTTON_6 5
#define MOUSE_BUTTON_7 6
#define MOUSE_BUTTON_8 7
#define MOUSE_BUTTON_LAST MOUSE_BUTTON_8
#define MOUSE_BUTTON_LEFT MOUSE_BUTTON_1
#define MOUSE_BUTTON_RIGHT MOUSE_BUTTON_2
#define MOUSE_BUTTON_MIDDLE MOUSE_BUTTON_3

#define KEY_SPACE 32
#define KEY_APOSTROPHE 39
#define KEY_COMMA 44
#define KEY_MINUS 45
#define KEY_PERIOD 46
#define KEY_SLASH 47
#define KEY_0 48
#define KEY_1 49
#define KEY_2 50
#define KEY_3 51
#define KEY_4 52
#define KEY_5 53
#define KEY_6 54
#define KEY_7 55
#define KEY_8 56
#define KEY_9 57
#define KEY_SEMICOLON 59
#define KEY_EQUAL 61
#define KEY_A 65
#define KEY_B 66
#define KEY_C 67
#define KEY_D 68
#define KEY_E 69
#define KEY_F 70
#define KEY_G 71
#define KEY_H 72
#define KEY_I 73
#define KEY_J 74
#define KEY_K 75
#define KEY_L 76
#define KEY_M 77
#define KEY_N 78
#define KEY_O 79
#define KEY_P 80
#define KEY_Q 81
#define KEY_R 82
#define KEY_S 83
#define KEY_T 84
#define KEY_U 85
#define KEY_V 86
#define KEY_W 87
#define KEY_X 88
#define KEY_Y 89
#define KEY_Z 90
#define KEY_LEFT_BRACKET 91
#define KEY_BACKSLASH 92
#define KEY_RIGHT_BRACKET 93
#define KEY_GRAVE_ACCENT 96
#define KEY_WORLD_1 161
#define KEY_WORLD_2 162
#define KEY_ESCAPE 256
#define KEY_ENTER 257
#define KEY_TAB 258
#define KEY_BACKSPACE 259
#define KEY_INSERT 260
#define KEY_DELETE 261
#define KEY_RIGHT 262
#define KEY_LEFT 263
#define KEY_DOWN 264
#define KEY_UP 265
#define KEY_PAGE_UP 266
#define KEY_PAGE_DOWN 267
#define KEY_HOME 268
#define KEY_END 269
#define KEY_CAPS_LOCK 280
#define KEY_SCROLL_LOCK 281
#define KEY_NUM_LOCK 282
#define KEY_PRINT_SCREEN 283
#define KEY_PAUSE 284
#define KEY_F1 290
#define KEY_F2 291
#define KEY_F3 292
#define KEY_F4 293
#define KEY_F5 294
#define KEY_F6 295
#define KEY_F7 296
#define KEY_F8 297
#define KEY_F9 298
#define KEY_F10 299
#define KEY_F11 300
#define KEY_F12 301
#define KEY_F13 302
#define KEY_F14 303
#define KEY_F15 304
#define KEY_F16 305
#define KEY_F17 306
#define KEY_F18 307
#define KEY_F19 308
#define KEY_F20 309
#define KEY_F21 310
#define KEY_F22 311
#define KEY_F23 312
#define KEY_F24 313
#define KEY_F25 314
#define KEY_KP_0 320
#define KEY_KP_1 321
#define KEY_KP_2 322
#define KEY_KP_3 323
#define KEY_KP_4 324
#define KEY_KP_5 325
#define KEY_KP_6 326
#define KEY_KP_7 327
#define KEY_KP_8 328
#define KEY_KP_9 329
#define KEY_KP_DECIMAL 330
#define KEY_KP_DIVIDE 331
#define KEY_KP_MULTIPLY 332
#define KEY_KP_SUBTRACT 333
#define KEY_KP_ADD 334
#define KEY_KP_ENTER 335
#define KEY_KP_EQUAL 336
#define KEY_LEFT_SHIFT 340
#define KEY_LEFT_CONTROL 341
#define KEY_LEFT_ALT 342
#define KEY_LEFT_SUPER 343
#define KEY_RIGHT_SHIFT 344
#define KEY_RIGHT_CONTROL 345
#define KEY_RIGHT_ALT 346
#define KEY_RIGHT_SUPER 347
#define KEY_MENU 348
#define KEY_LAST GLFW_KEY_MENU

// }}}

// {{{ OpenGL Versions

#define OPENGL_VER_1_0 10
#define OPENGL_VER_2_0 20
#define OPENGL_VER_3_0 30
#define OPENGL_VER_3_3 33
#define OPENGL_VER_4_0 40
#define OPENGL_VER_4_1 41
#define OPENGL_VER_4_2 42
#define OPENGL_VER_4_3 43
#define OPENGL_VER_4_4 44
#define OPENGL_VER_4_5 45
#define OPENGL_VER_4_6 46

// }}}

// {{{ Colors

typedef vec4f color;

#define RGB(r, g, b) (color){r, g, b, 255}
#define RGBA(r, g, b, a)                                                       \
    (color) { r, g, b, a }

#define RGB_NORM(r, g, b) (color){(r) * 255, (g) * 255, (b) * 255, 255}
#define RGBA_NORM(r, g, b, a)                                                  \
    (color){(r) * 255, (g) * 255, (b) * 255, (a) * 255}

#define WHITE RGB(255, 255, 255)
#define BLACK RGB(0, 0, 0)
#define RED RGB(255, 0, 0)
#define ORANGE RGB(255, 127, 0)
#define YELLOW RGB(255, 255, 0)
#define LIME_GREEN RGB(0, 255, 0)
#define GREEN RGB(0, 150, 25)
#define BLUE RGB(0, 0, 255)
#define LIGHT_BLUE RGB(0, 255, 255)
#define PURPLE RGB(127, 0, 255)
#define PINK RGB(255, 0, 255)
#define LIGHT_GRAY RGB(200, 200, 200)
#define DARK_GRAY RGB(64, 64, 64)
#define BROWN RGB(150, 76, 0)

// }}}

#define CPL_IMPLEMENTATION

// {{{ Logging

typedef enum { LOG_INFO, LOG_WARN, LOG_ERR, LOG_NONE } log_level;
void cpl_log(log_level level, c8 *msg, ...);

#ifdef CPL_IMPLEMENTATION
void cpl_log(log_level level, c8 *msg, ...) {
    va_list args;
    va_start(args, msg);
    switch (level) {
    case LOG_INFO:
        printf("[CPL] [INFO]: ");
        break;
    case LOG_WARN:
        printf("[CPL] [WARNING]: ");
        break;
    case LOG_ERR:
        fprintf(stderr, "[CPL] [ERROR]: ");
        break;
    case LOG_NONE:
        break;
    }
    while ((*msg) != '\0') {
        if ((*msg) == '%') {
            msg++;
            switch ((*msg)) {
            case 'c': {
                i32 c = va_arg(args, i32);
                putchar(c);
                break;
            }
            case 'i': {
                i32 i = va_arg(args, i32);
                if (i < 0) {
                    putchar('-');
                    i = -i;
                }
                if (i == 0) {
                    putchar('0');
                    break;
                }
                if (i / 10) {
                    cpl_log(LOG_NONE, "%i", i / 10);
                }
                putchar((i % 10) + '0');
                break;
            }
            case 's': {
                c8 *s = va_arg(args, c8 *);
                fputs(s, stdout);
                break;
            }
            default:
                break;
            }
        } else {
            putchar(*msg);
        }
        msg++;
    }
    va_end(args);
    printf("\n");
}
#endif

// }}}

// {{{ Screenshot

void cpl_screenshot(c8 *path, vec2f screen);

#ifdef CPL_IMPLEMENTATION
void cpl_screenshot(c8 *path, vec2f screen) {
    static i32 screenshots_taken = 0;
    i32 w = (i32)screen.x;
    i32 h = (i32)screen.y;
    i32 s = w * 3;
    u8 pixels[w * h * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    for (i32 y = 0; y < h / 2; y++) {
        for (i32 x = 0; x < s; x++) {
            u8 tmp = pixels[(y * s) + x];
            pixels[(y * s) + x] = pixels[((h - 1 - y) * s) + x];
            pixels[((h - 1 - y) * s) + x] = tmp;
        }
    }
    c8 final[100];
    snprintf(final, sizeof(final), "%sscreenshot%d.png", path,
             screenshots_taken);
    stbi_write_png(final, w, h, 3, pixels, s);
    screenshots_taken++;
}
#endif

// }}}

// {{{ Profiler

u32 cpl_get_heap_size();
u32 cpl_get_heap_used();
u32 cpl_get_heap_free();
u32 cpl_get_stack_size();
u32 cpl_get_stack_used();

#ifdef CPL_IMPLEMENTATION
u32 cpl_get_heap_size() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.arena;
#else
    return 0;
#endif
}

u32 cpl_get_heap_used() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.uordblks;
#else
    return 0;
#endif
}

u32 cpl_get_heap_free() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.fordblks;
#else
    return 0;
#endif
}

u32 cpl_get_stack_size() {
#ifndef __EMSCRIPTEN__
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    size_t size = 0;
    pthread_attr_getstacksize(&attr, &size);
    pthread_attr_destroy(&attr);
    return size;
#else
    return 0;
#endif
}

u32 cpl_get_stack_used() {
#ifndef __EMSCRIPTEN__
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    void *base = NULLPTR;
    size_t size = 0;
    pthread_attr_getstack(&attr, &base, &size);
    pthread_attr_destroy(&attr);
    c8 marker;
    void *cur = &marker;
    return (u32)(base + size - cur);
#else
    return 0;
#endif
}
#endif

// }}}

// {{{ OpenGL Debug

GLenum _cpl_check_gl_error(c8 *path, u32 line);
void cpl_check_opengl_error();
void cpl_enable_opengl_debug();
void APIENTRY _cpl_gl_debug_out(GLenum src, GLenum type, u32 id,
                                GLenum severity, [[maybe_unused]] GLsizei len,
                                const c8 *msg,
                                [[maybe_unused]] const void *usr_prog);

#ifdef CPL_IMPLEMENTATION
GLenum _cpl_check_gl_error(c8 *path, u32 line) {
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR) {
        c8 *error;
        switch (errorCode) {
        case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        default:
            error = "UNKNOWN";
            break;
        }
        fprintf(stderr, "[CPL] [ERROR] %s | %s (%d)\n", error, path, line);
    }
    return errorCode;
}

void cpl_check_opengl_error() { _cpl_check_gl_error(__FILE__, __LINE__); }

void APIENTRY _cpl_gl_debug_out(GLenum src, GLenum type, u32 id,
                                GLenum severity, [[maybe_unused]] GLsizei len,
                                const c8 *msg,
                                [[maybe_unused]] const void *usr_prog) {
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) {
        return;
    }

    fprintf(stderr, "[CPL] [ERROR](%d) %s\n", id, msg);

    fprintf(stderr, "->");
    switch (src) {
    case GL_DEBUG_SOURCE_API:
        fprintf(stderr, "Source: API");
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        fprintf(stderr, "Source: Window System");
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        fprintf(stderr, "Source: Shader Compiler");
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        fprintf(stderr, "Source: Third Party");
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        fprintf(stderr, "Source: Application");
        break;
    case GL_DEBUG_SOURCE_OTHER:
        fprintf(stderr, "Source: Other");
        break;
    default:
        fprintf(stderr, "Source: ???");
        break;
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "->");
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        fprintf(stderr, "Type: Error");
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        fprintf(stderr, "Type: Deprecated Behaviour");
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        fprintf(stderr, "Type: Undefined Behaviour");
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        fprintf(stderr, "Type: Portability");
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        fprintf(stderr, "Type: Performance");
        break;
    case GL_DEBUG_TYPE_MARKER:
        fprintf(stderr, "Type: Marker");
        break;
    case GL_DEBUG_TYPE_PUSH_GROUP:
        fprintf(stderr, "Type: Push Group");
        break;
    case GL_DEBUG_TYPE_POP_GROUP:
        fprintf(stderr, "Type: Pop Group");
        break;
    case GL_DEBUG_TYPE_OTHER:
        fprintf(stderr, "Type: Other");
        break;
    default:
        fprintf(stderr, "Type: ???");
        break;
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "->");
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        fprintf(stderr, "Severity: HIGH");
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        fprintf(stderr, "Severity: MEDIUM");
        break;
    case GL_DEBUG_SEVERITY_LOW:
        fprintf(stderr, "Severity: LOW");
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        fprintf(stderr, "Severity: NOTIFICTAION");
        break;
    default:
        fprintf(stderr, "Severity: ???");
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
}

void cpl_enable_opengl_debug() {
    i32 flags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);

    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
        GLint major = 0;
        GLint minor = 0;
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        if (major < 4 || (major == 4 && minor < 3)) {
            cpl_log(LOG_WARN, "OpenGL version is older than 4.3 - OpenGL debug "
                              "output disabled!");
            return;
        }

        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(_cpl_gl_debug_out, NULLPTR);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0,
                              NULLPTR, GL_TRUE);
    }
}
#endif

// }}}

// {{{ Shader

typedef struct {
    u32 id;
} shader;

b8 _cpl_check_shader_compile_errors(u32 shader, c8 *type);
c8 *_cpl_read_shader_file(c8 *path);
void cpl_create_shader(shader *s, c8 *vert_path, c8 *frag_path);
void cpl_use_shader(shader *s);
void cpl_shader_set_b8(shader *s, c8 *name, b8 val);
void cpl_shader_set_i32(shader *s, c8 *name, i32 val);
void cpl_shader_set_f32(shader *s, c8 *name, f32 val);
void cpl_shader_set_rgba(shader *s, c8 *name, vec4f c);
void cpl_shader_set_mat4f(shader *s, c8 *name, mat4f mat);
void cpl_shader_set_vec2f(shader *s, c8 *name, vec2f v);
void cpl_shader_set_vec3f(shader *s, c8 *name, vec3f v);

#ifdef CPL_IMPLEMENTATION
b8 _cpl_check_shader_compile_errors(u32 shader, c8 *type) {
    i32 success = 0;
    c8 info_log[1024];

    if (strcmp(type, "PROGRAM") == 0) {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULLPTR, info_log);
            cpl_log(LOG_ERR, "[CPL] [ERROR] Program linking error:\n%s\n",
                    info_log);
            return false;
        }
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULLPTR, info_log);
            cpl_log(LOG_ERR, "[CPL] [ERROR] Shader compilation error: %s\n%s\n",
                    type, info_log);
            return false;
        }
    }
    return true;
}

c8 *_cpl_read_shader_file(c8 *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULLPTR;
    }

    fseek(f, 0, SEEK_END);
    u32 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    c8 *buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULLPTR;
    }

    u32 read = fread(buffer, 1, size, f);
    fclose(f);

    if (read != size) {
        free(buffer);
        return NULLPTR;
    }

    buffer[size] = '\0';
    return buffer;
}

void cpl_create_shader(shader *s, c8 *vert_path, c8 *frag_path) {
    c8 *vert_code = _cpl_read_shader_file(vert_path);
    c8 *frag_code = _cpl_read_shader_file(frag_path);

    u32 vert = 0;
    u32 frag = 0;
    vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, (const GLchar *const *)&vert_code, NULLPTR);
    glCompileShader(vert);
    _cpl_check_shader_compile_errors(vert, "VERTEX");
    frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, (const GLchar *const *)&frag_code, NULLPTR);
    glCompileShader(frag);
    _cpl_check_shader_compile_errors(frag, "FRAGMENT");
    s->id = glCreateProgram();
    glAttachShader(s->id, vert);
    glAttachShader(s->id, frag);
    glLinkProgram(s->id);
    _cpl_check_shader_compile_errors(s->id, "PROGRAM");

    free(vert_code);
    free(frag_code);
    glDeleteShader(vert);
    glDeleteShader(frag);
}

void cpl_use_shader(shader *s) { glUseProgram(s->id); }

void cpl_shader_set_b8(shader *s, c8 *name, b8 val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_i32(shader *s, c8 *name, i32 val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_f32(shader *s, c8 *name, f32 val) {
    glUniform1f(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_rgba(shader *s, c8 *name, vec4f c) {
    glUniform4f(glGetUniformLocation(s->id, name), c.r / 255.0f, c.g / 255.0f,
                c.b / 255.0f, c.a / 255.0f);
}

void cpl_shader_set_mat4f(shader *s, c8 *name, mat4f mat) {
    glUniformMatrix4fv(glGetUniformLocation(s->id, name), 1, GL_FALSE,
                       (const GLfloat *)mat.data);
}

void cpl_shader_set_vec2f(shader *s, c8 *name, vec2f v) {
    glUniform2f(glGetUniformLocation(s->id, name), v.x, v.y);
}

void cpl_shader_set_vec3f(shader *s, c8 *name, vec3f v) {
    glUniform3f(glGetUniformLocation(s->id, name), v.x, v.y, v.z);
}
#endif

// }}}

// {{{ Rectangle

typedef struct {
    vec2f pos;
    vec2f size;
    vec4f color;
    f32 rot;
    u32 vbo, vao, ebo;
} rect;

void cpl_create_rect(rect *r, vec2f pos, vec2f size, vec4f color, f32 rot);
void cpl_destroy_rect(rect *r);
void cpl_draw_rect_raw(shader *s, rect *r);

#ifdef CPL_IMPLEMENTATION
void cpl_create_rect(rect *r, vec2f pos, vec2f size, vec4f color, f32 rot) {
    r->pos = pos;
    r->size = size;
    r->color = color;
    r->rot = rot;

    f32 vertices[12] = {
        size.x, 0.0f,   0.0f, //
        size.x, size.y, 0.0f, //
        0.0f,   size.y, 0.0f, //
        0.0f,   0.0f,   0.0f  //
    };
    u32 indices[6] = {
        0, 1, 3, //
        1, 2, 3  //
    };

    glGenVertexArrays(1, &r->vao);
    glGenBuffers(1, &r->vbo);
    glGenBuffers(1, &r->ebo);
    glBindVertexArray(r->vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void cpl_destroy_rect(rect *r) {
    if (r->vao != 0 && glIsVertexArray(r->vao)) {
        glDeleteVertexArrays(1, &r->vao);
        r->vao = 0;
    }
    if (r->vbo != 0 && glIsBuffer(r->vbo)) {
        glDeleteBuffers(1, &r->vbo);
        r->vbo = 0;
    }
    if (r->ebo != 0 && glIsBuffer(r->ebo)) {
        glDeleteBuffers(1, &r->ebo);
        r->ebo = 0;
    }
}

void cpl_draw_rect_raw(shader *s, rect *r) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, &(vec3f){r->pos.x, r->pos.y, 0.0f});
    mat4f_translate(&transform,
                    &(vec3f){r->size.x * 0.5f, r->size.y * 0.5f, 0.0f});
    mat4f_rotate(&transform, cpm_rad(r->rot), &(vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(&transform,
                    &(vec3f){-r->size.x * 0.5f, -r->size.y * 0.5f, 0.0f});

    cpl_shader_set_mat4f(s, "transform", transform);
    cpl_shader_set_rgba(s, "input_color", r->color);

    glBindVertexArray(r->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULLPTR);
    glBindVertexArray(0);
}
#endif

// }}}

// {{{ Triangle

typedef struct {
    vec2f pos;
    vec2f size;
    vec4f color;
    f32 rot;
    u32 vbo, vao;
} triangle;

void cpl_create_triangle(triangle *t, vec2f pos, vec2f size, vec4f color,
                         f32 rot);
void cpl_destroy_triangle(triangle *t);
void cpl_draw_triangle_raw(shader *s, triangle *t);

#ifdef CPL_IMPLEMENTATION
void cpl_create_triangle(triangle *t, vec2f pos, vec2f size, vec4f color,
                         f32 rot) {
    t->pos = pos;
    t->size = size;
    t->color = color;
    t->rot = rot;

    f32 vertices[9] = {
        0.0f,          0.0f,   0.0f, //
        size.x,        0.0f,   0.0f, //
        size.x / 2.0f, size.y, 0.0f  //
    };

    glGenVertexArrays(1, &t->vao);
    glGenBuffers(1, &t->vbo);
    glBindVertexArray(t->vao);
    glBindBuffer(GL_ARRAY_BUFFER, t->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void cpl_destroy_triangle(triangle *t) {
    if (t->vao != 0 && glIsVertexArray(t->vao)) {
        glDeleteVertexArrays(1, &t->vao);
        t->vao = 0;
    }
    if (t->vbo != 0 && glIsBuffer(t->vbo)) {
        glDeleteBuffers(1, &t->vbo);
        t->vbo = 0;
    }
}

void cpl_draw_triangle_raw(shader *s, triangle *t) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, &(vec3f){t->pos.x, t->pos.y, 0.0f});
    mat4f_translate(&transform,
                    &(vec3f){t->size.x * 0.5f, t->size.y * 0.5f, 0.0f});
    mat4f_rotate(&transform, cpm_rad(t->rot), &(vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(&transform,
                    &(vec3f){-t->size.x * 0.5f, -t->size.y * 0.5f, 0.0f});

    cpl_shader_set_mat4f(s, "transform", transform);
    cpl_shader_set_rgba(s, "input_color", t->color);

    glBindVertexArray(t->vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}
#endif

// }}}

// {{{ Circle

typedef struct {
    vec2f pos;
    f32 radius;
    vec4f color;
    u32 vao, vbo;
    i32 vertex_cnt;
} circle;

void cpl_create_circle(circle *c, vec2f pos, f32 radius, vec4f color);
void cpl_destroy_circle(circle *c);
void cpl_draw_circle_raw(shader *s, circle *c);

#ifdef CPL_IMPLEMENTATION
void cpl_create_circle(circle *c, vec2f pos, f32 radius, vec4f color) {
    c->pos = pos;
    c->color = color;
    c->radius = radius;

    i32 segments = CPM_MIN(32, (i32)cpm_ceilf(2.0f * CPM_PI * radius / 2.0f));
    u64 vertices_size = ((u64)(segments + 1) * 3) + 3;
    f32 *vertices = malloc(vertices_size * sizeof(f32));
    for (u32 i = 0; i < 3; i++) {
        vertices[i] = 0;
    }
    for (u32 i = 0; i <= segments; i++) {
        f32 theta = 2 * CPM_PI / (f32)segments * (f32)i;
        f32 x = 0.0f + (radius * cpm_cosf(theta));
        f32 y = 0.0f + (radius * cpm_sinf(theta));
        vertices[(u64)(i + 1) * 3] = x;
        vertices[((i + 1) * 3) + 1] = y;
        vertices[((i + 1) * 3) + 2] = 0.0f;
    }
    c->vertex_cnt = (i32)(vertices_size / 3);
    glGenVertexArrays(1, &c->vao);
    glGenBuffers(1, &c->vbo);
    glBindVertexArray(c->vao);
    glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
    glBufferData(GL_ARRAY_BUFFER, (i32)(vertices_size * sizeof(f32)), vertices,
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    free(vertices);
}

void cpl_destroy_circle(circle *c) {
    if (c->vao != 0 && glIsVertexArray(c->vao)) {
        glDeleteVertexArrays(1, &c->vao);
        c->vao = 0;
    }
    if (c->vbo != 0 && glIsBuffer(c->vbo)) {
        glDeleteBuffers(1, &c->vbo);
        c->vbo = 0;
    }
}

void cpl_draw_circle_raw(shader *s, circle *c) {
    mat4f transform;
    mat4f_identity(&transform);
    mat4f_translate(&transform, &(vec3f){c->pos.x, c->pos.y, 0.0f});

    cpl_shader_set_mat4f(s, "transform", transform);
    cpl_shader_set_rgba(s, "input_color", c->color);

    glBindVertexArray(c->vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, c->vertex_cnt);
    glBindVertexArray(0);
}
#endif

// }}}

// {{{ Line

typedef struct {
    vec2f start, end;
    vec4f color;
    u32 vao, vbo;
} line;

void cpl_create_line(line *l, vec2f start, vec2f end, vec4f color);
void cpl_destroy_line(line *l);
void cpl_draw_line_raw(shader *s, line *l);

#ifdef CPL_IMPLEMENTATION
void cpl_create_line(line *l, vec2f start, vec2f end, vec4f color) {
    l->start = start;
    l->end = end;
    l->color = color;

    f32 vertices[6] = {
        start.x, start.y, 0.0f, //
        end.x,   end.y,   0.0f, //
    };

    glGenVertexArrays(1, &l->vao);
    glGenBuffers(1, &l->vbo);
    glBindVertexArray(l->vao);
    glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
void cpl_destroy_line(line *l) {
    if (l->vao != 0 && glIsVertexArray(l->vao)) {
        glDeleteVertexArrays(1, &l->vao);
        l->vao = 0;
    }
    if (l->vbo != 0 && glIsBuffer(l->vbo)) {
        glDeleteBuffers(1, &l->vbo);
        l->vbo = 0;
    }
}

void cpl_draw_line_raw(shader *s, line *l) {
    mat4f transform;
    mat4f_identity(&transform);
    mat4f_translate(&transform, &(vec3f){0.0f, 0.0f, 0.0f});

    cpl_shader_set_mat4f(s, "transform", transform);
    cpl_shader_set_rgba(s, "input_color", l->color);
    glBindVertexArray(l->vao);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
}
#endif

// }}}

// {{{ Texture & Texture2D

typedef enum { FILTER_LINEAR, FILTER_NEAREST } texture_filtering;

typedef struct {
    u32 id;
    vec2f size;
} texture;
typedef struct {
    vec2f pos;
    vec2f size;
    f32 rot;
    vec4f color;
    texture *tex;
    u32 vbo, vao, ebo;
} texture2D;

void cpl_load_texture(texture *t, c8 *path, texture_filtering filter);
void cpl_unload_texture(texture *t);
void cpl_create_texture2D(texture2D *t, vec2f pos, vec2f size, f32 rot,
                          vec4f color, texture *tex);
void cpl_destroy_texture2D(texture2D *t);
void cpl_draw_texture2D_raw(shader *s, texture2D *t);

#ifdef CPL_IMPLEMENTATION
void cpl_load_texture(texture *t, c8 *path, texture_filtering filter) {
    glGenTextures(1, &t->id);
    glBindTexture(GL_TEXTURE_2D, t->id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
    stbi_set_flip_vertically_on_load(1);
    i32 width = 0;
    i32 height = 0;
    i32 channels = 0;
    u8 *data = stbi_load(path, &width, &height, &channels, 0);
    GLenum format = 0;
    if (channels == 1) {
        format = GL_RED;
    } else if (channels == 3) {
        format = GL_RGB;
    } else if (channels == 4) {
        format = GL_RGBA;
    }
    if (data) {
        t->size.x = (f32)width;
        t->size.y = (f32)height;
        glTexImage2D(GL_TEXTURE_2D, 0, (i32)format, (i32)t->size.x,
                     (i32)t->size.y, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        cpl_log(LOG_ERR, "Failed to load texture");
    }
    stbi_image_free(data);
}

void cpl_unload_texture(texture *t) {
    if (t->id != 0) {
        glDeleteTextures(1, &t->id);
    }
}

void cpl_create_texture2D(texture2D *t, vec2f pos, vec2f size, f32 rot,
                          vec4f color, texture *tex) {
    t->pos = pos;
    t->size = size;
    t->rot = rot;
    t->color = color;
    t->tex = tex;

    f32 vertices[22] = {
        size.x, 0.0f,   0.0f, 1.0f, 1.0f, //
        size.x, size.y, 0.0f, 1.0f, 0.0f, //
        0.0f,   size.y, 0.0f, 0.0f, 0.0f, //
        0.0f,   0.0f,   0.0f, 0.0f, 1.0f  //
    };
    u32 indices[6] = {
        0, 1, 3, //
        1, 2, 3  //
    };

    glGenVertexArrays(1, &t->vao);
    glGenBuffers(1, &t->vbo);
    glGenBuffers(1, &t->ebo);
    glBindVertexArray(t->vao);
    glBindBuffer(GL_ARRAY_BUFFER, t->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                 GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                          (void *)(3 * sizeof(f32)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void cpl_destroy_texture2D(texture2D *t) {
    if (t->vao != 0 && glIsVertexArray(t->vao)) {
        glDeleteVertexArrays(1, &t->vao);
        t->vao = 0;
    }
    if (t->vbo != 0 && glIsBuffer(t->vbo)) {
        glDeleteBuffers(1, &t->vbo);
        t->vbo = 0;
    }
    if (t->ebo != 0 && glIsBuffer(t->ebo)) {
        glDeleteBuffers(1, &t->ebo);
        t->ebo = 0;
    }
    t->tex = NULLPTR;
}

void cpl_draw_texture2D_raw(shader *s, texture2D *t) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, &(vec3f){t->pos.x, t->pos.y, 0.0f});
    mat4f_translate(&transform,
                    &(vec3f){t->size.x * 0.5f, t->size.y * 0.5f, 0.0f});
    mat4f_rotate(&transform, cpm_rad(t->rot), &(vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(&transform,
                    &(vec3f){-t->size.x * 0.5f, -t->size.y * 0.5f, 0.0f});

    cpl_shader_set_i32(s, "tex", 0);
    cpl_shader_set_mat4f(s, "transform", transform);
    cpl_shader_set_rgba(s, "input_color", t->color);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t->tex->id);
    glBindVertexArray(t->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULLPTR);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
#endif

// }}}

// {{{ Text

typedef struct {
    u32 id;
    vec2f size;
    vec2f bearing;
    u32 advance;
} letter;
VEC_DEF(letter, vec_letters)
typedef struct {
    u32 vao, vbo;
    c8 *name;
    vec_letters letters;
} font;

void cpl_create_font(font *f, c8 *path, c8 *name, texture_filtering filter);
void cpl_delete_font(font *f);
void cpl_draw_text_raw(shader *s, font *f, c8 *text, vec2f pos, f32 scale,
                       vec4f color);
vec2f cpl_get_text_size(font *f, c8 *text, f32 scale);

#ifdef CPL_IMPLEMENTATION
void cpl_create_font(font *f, c8 *path, c8 *name, texture_filtering filter) {
    FT_Library ft;
    if (FT_Init_FreeType(&ft)) {
        cpl_log(LOG_ERR, "Could not init FreeType Library");
        exit(-1);
    }

    if (access(path, F_OK) == -1) {
        cpl_log(LOG_ERR, "Failed to load %s", name);
        exit(-1);
    }

    FT_Face face;
    if (FT_New_Face(ft, path, 0, &face)) {
        cpl_log(LOG_ERR, "Failed to load font");
        exit(-1);
    }
    FT_Set_Pixel_Sizes(face, 0, 48);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    vec_letters_reserve(&f->letters, 128);
    for (u8 c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            cpl_log(LOG_ERR, "Failed to load Glyph");
            continue;
        }

        u32 tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
                     (GLsizei)face->glyph->bitmap.width,
                     (GLsizei)face->glyph->bitmap.rows, 0, GL_RED,
                     GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                        filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);

        letter character = {.id = tex,
                            .size = {(f32)face->glyph->bitmap.width,
                                     (f32)face->glyph->bitmap.rows},
                            .bearing = {(f32)face->glyph->bitmap_left,
                                        (f32)face->glyph->bitmap_top},
                            .advance = face->glyph->advance.x};
        vec_letters_push_back(&f->letters, character);
    }
    f->name = name;

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glGenVertexArrays(1, &f->vao);
    glGenBuffers(1, &f->vbo);
    glBindVertexArray(f->vao);
    glBindBuffer(GL_ARRAY_BUFFER, f->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(f32) * 6 * 4, NULLPTR,
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(f32), NULLPTR);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    FT_Done_Face(face);
    FT_Done_FreeType(ft);
}

void cpl_delete_font(font *f) {
    if (f->vao != 0 && glIsVertexArray(f->vao)) {
        glDeleteVertexArrays(1, &f->vao);
        f->vao = 0;
    }
    if (f->vbo != 0 && glIsBuffer(f->vbo)) {
        glDeleteBuffers(1, &f->vbo);
        f->vbo = 0;
    }
    for (u32 i = 0; i < f->letters.size; i++) {
        glDeleteTextures(1, &vec_letters_at(&f->letters, i)->id);
    }
    vec_letters_destroy(&f->letters);
}

void cpl_draw_text_raw(shader *s, font *f, c8 *text, vec2f pos, f32 scale,
                       vec4f color) {
    cpl_shader_set_vec3f(s, "text_color", (vec3f){color.r, color.g, color.b});
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(f->vao);

    for (u32 i = 0; i < strlen(text); i++) {
        letter *l = vec_letters_at(&f->letters, text[i]);

        f32 x_pos = pos.x + (l->bearing.x * scale);
        f32 y_pos =
            pos.y +
            ((vec_letters_at(&f->letters, 'H')->bearing.y - l->bearing.y) *
             scale);
        f32 width = l->size.x * scale;
        f32 height = l->size.y * scale;

        f32 vertices[6][4] = {{x_pos, y_pos + height, 0.0f, 1.0f},
                              {x_pos, y_pos, 0.0f, 0.0f},
                              {x_pos + width, y_pos, 1.0f, 0.0f},

                              {x_pos, y_pos + height, 0.0f, 1.0f},
                              {x_pos + width, y_pos, 1.0f, 0.0f},
                              {x_pos + width, y_pos + height, 1.0f, 1.0f}};

        glBindTexture(GL_TEXTURE_2D, l->id);
        glBindBuffer(GL_ARRAY_BUFFER, f->vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        pos.x += ((f32)(l->advance >> 6)) * scale;
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

vec2f cpl_get_text_size(font *f, c8 *text, f32 scale) {
    f32 width = 0.0f;
    f32 height = 0.0f;
    f32 max_above_base = 0.0f;
    f32 max_below_base = 0.0f;

    for (u32 i = 0; i < strlen(text); i++) {

        letter *l = vec_letters_at(&f->letters, text[i]);
        f32 h = l->size.y * scale;
        max_above_base = CPM_MAX(max_above_base, l->bearing.y * scale);
        max_below_base = CPM_MAX(max_below_base, (h - (l->bearing.y * scale)));
        width += (f32)(l->advance >> 6) * scale;
    }
    height = max_above_base + max_below_base;
    return (vec2f){width, height};
}
#endif

// }}}

// {{{ Audio

typedef struct {
    c8 *path;
    f32 volume;
    f32 pitch;
} audio;

ma_engine _cpl_audio_engine;
ma_sound *_cpl_music;
ma_sound **_cpl_active_sounds;
u32 _cpl_active_sounds_size;
u32 _cpl_active_sounds_cap;

void cpl_audio_init();
audio cpl_load_audio(c8 *path);
void cpl_audio_update();
void cpl_audio_play_sound(audio *a);
void cpl_audio_play_music(audio *a);
void cpl_audio_pause_music();
void cpl_audio_resume_music();
void cpl_audio_stop_music();
void cpl_audio_close();

#ifdef CPL_IMPLEMENTATION
void cpl_audio_init() {
    if (ma_engine_init(NULLPTR, &_cpl_audio_engine) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to init audio!");
        exit(-1);
    }
    _cpl_active_sounds_cap = 16;
    _cpl_active_sounds_size = 0;
    _cpl_active_sounds = malloc(_cpl_active_sounds_cap * sizeof(ma_sound *));
    _cpl_music = NULLPTR;
}

audio cpl_load_audio(c8 *path) { return (audio){path, 1.0f, 1.0f}; }

void cpl_audio_update() {
    u32 w = 0;
    for (u32 i = 0; i < _cpl_active_sounds_size; i++) {
        if (ma_sound_is_playing(_cpl_active_sounds[i])) {
            _cpl_active_sounds[w++] = _cpl_active_sounds[i];
        } else {
            ma_sound_uninit(_cpl_active_sounds[i]);
            free(_cpl_active_sounds[i]);
        }
    }
    _cpl_active_sounds_size = w;
}

void cpl_audio_play_sound(audio *a) {
    ma_sound *sound = malloc(sizeof(ma_sound));
    if (ma_sound_init_from_file(&_cpl_audio_engine, a->path,
                                MA_SOUND_FLAG_DECODE, NULLPTR, NULLPTR,
                                sound) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to init sound!");
        free(sound);
        return;
    }
    ma_sound_set_pitch(sound, a->pitch);
    ma_sound_set_volume(sound, a->volume);
    ma_sound_set_looping(sound, MA_FALSE);
    ma_sound_start(sound);

    if (_cpl_active_sounds_size >= _cpl_active_sounds_cap) {
        _cpl_active_sounds_cap *= 2;
        ma_sound **tmp = realloc(_cpl_active_sounds,
                                 _cpl_active_sounds_cap * sizeof(ma_sound *));
        if (!tmp) {
            cpl_log(LOG_ERR, "Realloc sounds failed!");
            return;
        }
        _cpl_active_sounds = tmp;
    }
    _cpl_active_sounds[_cpl_active_sounds_size++] = sound;
}

void cpl_audio_play_music(audio *a) {
    if (_cpl_music) {
        ma_sound_stop(_cpl_music);
        ma_sound_uninit(_cpl_music);
        free(_cpl_music);
        _cpl_music = NULLPTR;
    }
    _cpl_music = malloc(sizeof(ma_sound));
    if (ma_sound_init_from_file(&_cpl_audio_engine, a->path,
                                MA_SOUND_FLAG_DECODE, NULLPTR, NULLPTR,
                                _cpl_music) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to load music!");
        free(_cpl_music);
        _cpl_music = NULLPTR;
        return;
    }
    ma_sound_set_pitch(_cpl_music, a->pitch);
    ma_sound_set_looping(_cpl_music, MA_TRUE);
    ma_sound_start(_cpl_music);
}

void cpl_audio_pause_music() {
    if (_cpl_music) {
        ma_sound_stop(_cpl_music);
    }
}

void cpl_audio_resume_music() {
    if (_cpl_music) {
        ma_sound_start(_cpl_music);
    }
}

void cpl_audio_stop_music() {
    if (_cpl_music) {
        ma_sound_stop(_cpl_music);
        ma_sound_seek_to_pcm_frame(_cpl_music, 0);
    }
}

void cpl_audio_close() {
    cpl_audio_update();
    free(_cpl_active_sounds);
    if (_cpl_music) {
        ma_sound_stop(_cpl_music);
        ma_sound_uninit(_cpl_music);
        free(_cpl_music);
    }
    ma_engine_uninit(&_cpl_audio_engine);
}
#endif

// }}}

// {{{ Screen Quad

typedef struct {
    vec2f size;
    u32 vbo, vao, rbo, framebuffer, tex_color_buffer;
} screen_quad;

void cpl_create_screen_quad(screen_quad *q, i32 width, i32 height);
void cpl_screen_quad_resize(screen_quad *q, i32 width, i32 height);
void cpl_screen_quad_bind(screen_quad *q);
void cpl_screen_quad_unbind();
void cpl_screen_quad_draw(screen_quad *q, shader *s);

#ifdef CPL_IMPLEMENTATION
void cpl_create_screen_quad(screen_quad *q, i32 width, i32 height) {
    q->size = (vec2f){(f32)width, (f32)height};

    f32 vertices[30] = {-1.0f, 1.0f, 0.0f, 0.0f,  1.0f, -1.0f, -1.0f, 0.0f,
                        0.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  0.0f,

                        -1.0f, 1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  -1.0f, 0.0f,
                        1.0f,  0.0f, 1.0f, 1.0f,  0.0f, 1.0f,  1.0f};

    glGenVertexArrays(1, &q->vao);
    glGenBuffers(1, &q->vbo);
    glBindVertexArray(q->vao);
    glBindBuffer(GL_ARRAY_BUFFER, q->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                          (void *)NULLPTR);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                          (void *)(3 * sizeof(f32)));

    glGenFramebuffers(1, &q->framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, q->framebuffer);

    glGenTextures(1, &q->tex_color_buffer);
    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (i32)q->size.x, (i32)q->size.y, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, NULLPTR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           q->tex_color_buffer, 0);

    glGenRenderbuffers(1, &q->rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, q->rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (i32)q->size.x,
                          (i32)q->size.y);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                              GL_RENDERBUFFER, q->rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cpl_log(LOG_ERR, "Framebuffer is not complete!");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void cpl_screen_quad_resize(screen_quad *q, i32 width, i32 height) {
    q->size = (vec2f){(f32)width, (f32)height};

    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, NULLPTR);

    glBindRenderbuffer(GL_RENDERBUFFER, q->rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void cpl_screen_quad_bind(screen_quad *q) {
    glBindFramebuffer(GL_FRAMEBUFFER, q->framebuffer);
}
void cpl_screen_quad_unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void cpl_screen_quad_draw(screen_quad *q, shader *s) {
    cpl_use_shader(s);
    glBindVertexArray(q->vao);
    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}
#endif

// }}}

// {{{ Collisions

typedef struct {
    vec2f pos;
    vec2f size;
} rect_collider;
typedef struct {
    vec2f pos;
    vec2f size;
} triangle_collider;
typedef struct {
    vec2f pos;
    f32 radius;
} circle_collider;

b8 cpl_check_collision_rects(rect_collider a, rect_collider b);
b8 cpl_check_collision_circle_rect(circle_collider a, rect_collider b);
b8 cpl_check_collision_vec2f_rect(vec2f a, rect_collider b);
b8 cpl_check_collision_circles(circle_collider a, circle_collider b);
b8 cpl_check_collision_vec2f_circle(vec2f a, circle_collider b);

#ifdef CPL_IMPLEMENTATION
b8 cpl_check_collision_rects(rect_collider a, rect_collider b) {
    b8 collision_x =
        a.pos.x + a.size.x >= b.pos.x && b.pos.x + b.size.x >= a.pos.x;
    b8 collision_y =
        a.pos.y + a.size.y >= b.pos.y && b.pos.y + b.size.y >= a.pos.y;

    return collision_x && collision_y;
}

b8 cpl_check_collision_circle_rect(circle_collider a, rect_collider b) {
    vec2f circle_center = a.pos;
    vec2f rect_center =
        vec2f_add(&b.pos, &VEC2F(b.size.x * 0.5f, b.size.y * 0.5f));
    vec2f half_extents = VEC2F(b.size.x * 0.5f, b.size.y * 0.5f);
    vec2f difference = vec2f_sub(&circle_center, &rect_center);
    vec2f clamped = vec2f_clamp(
        &difference, &VEC2F(-half_extents.x, -half_extents.y), &half_extents);
    vec2f closest = vec2f_add(&rect_center, &clamped);
    vec2f delta = vec2f_sub(&closest, &circle_center);

    return vec2f_length(&delta) <= a.radius;
}

b8 cpl_check_collision_vec2f_rect(vec2f a, rect_collider b) {
    return b.pos.x < a.x && a.x < b.pos.x + b.size.x && b.pos.y < a.y &&
           a.y < b.pos.y + b.size.y;
}

b8 cpl_check_collision_circles(circle_collider a, circle_collider b) {
    vec2f dist = vec2f_sub(&a.pos, &b.pos);
    f32 distance2 = (dist.x * dist.x) + (dist.y * dist.y);
    f32 radius_sum = a.radius + b.radius;
    return distance2 <= radius_sum * radius_sum;
}

b8 cpl_check_collision_vec2f_circle(vec2f a, circle_collider b) {
    vec2f dist = vec2f_sub(&a, &b.pos);
    f32 distance2 = (dist.x * dist.x) + (dist.y * dist.y);
    return distance2 <= b.radius * b.radius;
}
#endif

// }}}

// {{{ Timing

u32 _cpl_nb_frames = 0;
f32 _cpl_last_time = 0.0f;
f32 _cpl_last_frame = 0.0f;
f32 _cpl_dt = 0.0f;
f32 _cpl_time_scale = 1.0f;
u32 _cpl_fps = 0;

void cpl_calc_fps();
u32 cpl_get_fps();
void cpl_calc_dt();
f32 cpl_get_dt();
f32 cpl_get_time();
f32 cpl_get_time_scale();
void cpl_set_time_scale(f32 scale);

#ifdef CPL_IMPLEMENTATION
void cpl_calc_fps() {
    f32 cur_time = cpl_get_time();
    _cpl_nb_frames++;
    if (cur_time - _cpl_last_time >= 1.0) {
        _cpl_fps = _cpl_nb_frames;
        _cpl_nb_frames = 0;
        _cpl_last_time += 1.0f;
    }
}
u32 cpl_get_fps() { return _cpl_fps; }

void cpl_calc_dt() {
    f32 cur_frame = cpl_get_time();
    _cpl_dt = (cur_frame - _cpl_last_frame) * _cpl_time_scale;
    _cpl_last_frame = cur_frame;
}

f32 cpl_get_dt() { return _cpl_dt; }
f32 cpl_get_time() {
    static struct timespec start_ts;
    static b8 initialized = false;
    struct timespec cur_ts;
    if (!initialized) {
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
        initialized = true;
    }
    clock_gettime(CLOCK_MONOTONIC, &cur_ts);
    return (f32)((f64)(cur_ts.tv_sec - start_ts.tv_sec) +
                 ((f64)(cur_ts.tv_nsec - start_ts.tv_nsec) * 1e-9));
}
f32 cpl_get_time_scale() { return _cpl_time_scale; }

void cpl_set_time_scale(f32 scale) { _cpl_time_scale = scale; }
#endif

// }}}

// {{{ Inputs

typedef struct {
    vec2f pos;
    f32 zoom;
    f32 rot;
} cam_2D;

GLFWwindow *_cpl_window = NULLPTR;
cam_2D _cpl_cam_2D;

b8 _cpl_key_states[KEY_LAST - KEY_SPACE + 1];
b8 _cpl_prev_key_states[KEY_LAST - KEY_SPACE + 1];
b8 _cpl_mouse_button_states[MOUSE_BUTTON_LAST + 1];
b8 _cpl_prev_mouse_button_states[MOUSE_BUTTON_LAST + 1];

void cpl_update_input();
b8 cpl_is_key_down(i32 key);
b8 cpl_is_key_up(i32 key);
b8 cpl_is_key_pressed(i32 key);
b8 cpl_is_key_released(i32 key);
b8 cpl_is_mouse_down(i32 button);
b8 cpl_is_mouse_pressed(i32 button);
b8 cpl_is_mouse_released(i32 button);
mat4f *cpl_cam_2D_get_view_mat(cam_2D *cam);
cam_2D *cpl_get_cam_2D();
vec2f cpl_get_mouse_pos();
vec2f cpl_get_screen_to_world_2D(vec2f sp);

#ifdef CPL_IMPLEMENTATION
void cpl_update_input() {
    for (u32 i = 0; i < KEY_LAST - KEY_SPACE; i++) {
        _cpl_prev_key_states[i] = _cpl_key_states[i];
    }
    for (u32 key = KEY_SPACE; key <= KEY_LAST; key++) {
        _cpl_key_states[key - KEY_SPACE] =
            glfwGetKey(_cpl_window, (i32)key) == GLFW_PRESS;
    }

    for (u32 i = 0; i < MOUSE_BUTTON_LAST - MOUSE_BUTTON_1; i++) {
        _cpl_prev_mouse_button_states[i] = _cpl_mouse_button_states[i];
    }
    for (u32 button = MOUSE_BUTTON_1; button <= MOUSE_BUTTON_LAST; button++) {
        _cpl_mouse_button_states[button - MOUSE_BUTTON_1] =
            glfwGetMouseButton(_cpl_window, (i32)button) == GLFW_PRESS;
    }
}

b8 cpl_is_key_down(i32 key) { return _cpl_key_states[key - KEY_SPACE]; }
b8 cpl_is_key_up(i32 key) { return !_cpl_key_states[key - KEY_SPACE]; }
b8 cpl_is_key_pressed(i32 key) {
    return _cpl_key_states[key - KEY_SPACE] &&
           !_cpl_prev_key_states[key - KEY_SPACE];
}
b8 cpl_is_key_released(i32 key) {
    return !_cpl_key_states[key - KEY_SPACE] &&
           _cpl_prev_key_states[key - KEY_SPACE];
}

b8 cpl_is_mouse_down(i32 button) {
    return _cpl_mouse_button_states[button - MOUSE_BUTTON_1];
}
b8 cpl_is_mouse_pressed(i32 button) {
    return _cpl_mouse_button_states[button - MOUSE_BUTTON_1] &&
           !_cpl_prev_mouse_button_states[button - MOUSE_BUTTON_1];
}
b8 cpl_is_mouse_released(i32 button) {
    return !_cpl_mouse_button_states[button - MOUSE_BUTTON_1] &&
           _cpl_prev_mouse_button_states[button - MOUSE_BUTTON_1];
}

mat4f *cpl_cam_2D_get_view_mat(cam_2D *cam) {
    mat4f *view = malloc(sizeof(mat4f));
    mat4f_identity(view);

    mat4f_translate(view, &(vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    mat4f_translate(view, &(vec3f){cam->pos.x, cam->pos.y, 0.0f});
    mat4f_rotate(view, cpm_rad(cam->rot), &(vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(view, &(vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    mat4f_translate(view, &(vec3f){cam->pos.x, cam->pos.y, 0.0f});
    mat4f_scale(view, &(vec3f){cam->zoom, cam->zoom, 1.0f});
    mat4f_translate(view, &(vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    return view;
}

cam_2D *cpl_get_cam_2D() { return &_cpl_cam_2D; }

vec2f cpl_get_mouse_pos() {
    f64 x = 0;
    f64 y = 0;
    glfwGetCursorPos(_cpl_window, &x, &y);
    return VEC2F((f32)x, (f32)y);
}
vec2f cpl_get_screen_to_world_2D(vec2f sp) {
    f32 x = sp.x;
    f32 y = sp.y;
    x /= _cpl_cam_2D.zoom;
    y /= _cpl_cam_2D.zoom;
    x += _cpl_cam_2D.pos.x;
    y += _cpl_cam_2D.pos.y;
    return (vec2f){x, y};
}
#endif

// }}}

// TODO rewrite / correct from here

// {{{ Window

typedef enum {
    SHAPE_2D_UNLIT,
    SHAPE_2D_LIT,
    TEXT,
    TEXTURE_2D_UNLIT,
    TEXTURE_2D_LIT,
    _DRAW_MODES_COUNT
} cpl_draw_mode;

GLubyte *_cpl_renderer;
GLubyte *_cpl_vendor;
GLubyte *_cpl_version;

u32 _cpl_screen_width = 0;
u32 _cpl_screen_height = 0;
mat4f _cpl_projection_2D;
cpl_draw_mode _cpl_cur_draw_mode = SHAPE_2D_UNLIT;
shader _cpl_shaders[_DRAW_MODES_COUNT];

void _cpl_framebuffer_size_callback([[maybe_unused]] GLFWwindow *window,
                                    i32 width, i32 height);
void _cpl_web_window_resize();
void _cpl_init_shaders();
void cpl_init_window(u32 width, u32 height, c8 *title, u32 version);
b8 cpl_window_should_close();
void cpl_destroy_window();
void cpl_close_window();
u32 cpl_get_screen_width();
u32 cpl_get_screen_height();
vec2f cpl_get_screen_size();
void cpl_enable_vsync(b8 enabled);
void cpl_update();
void cpl_end_frame();

#ifdef CPL_IMPLEMENTATION
void _cpl_framebuffer_size_callback([[maybe_unused]] GLFWwindow *window,
                                    i32 width, i32 height) {
    glViewport(0, 0, width, height);
    _cpl_screen_width = width;
    _cpl_screen_height = height;
    mat4f_ortho(&_cpl_projection_2D, 0.0f, (f32)_cpl_screen_width,
                (f32)_cpl_screen_height, 0.0f, -1.0f, 1.0f);
}

void _cpl_web_window_resize() {
#ifdef __EMSCRIPTEN__
    i32 w = emscripten_run_script_int("window.innerWidth");
    i32 h = emscripten_run_script_int("window.innerHeight");
    if ((f32)w != cpl_screen_width || (f32)h != cpl_screen_height) {
        glfwSetWindowSize(cpl_window, w, h);
    }
#endif
}

void _cpl_init_shaders() {
#ifndef __EMSCRIPTEN__
    cpl_create_shader(&_cpl_shaders[SHAPE_2D_UNLIT],
                      "shaders/vert/2D/shape.vert",
                      "shaders/frag/2D/shape_unlit.frag");
    cpl_create_shader(&_cpl_shaders[SHAPE_2D_LIT], "shaders/vert/2D/shape.vert",
                      "shaders/frag/2D/shape_lit.frag");
    cpl_create_shader(&_cpl_shaders[TEXT], "shaders/vert/2D/text.vert",
                      "shaders/frag/2D/text.frag");
    cpl_create_shader(&_cpl_shaders[TEXTURE_2D_UNLIT],
                      "shaders/vert/2D/texture.vert",
                      "shaders/frag/2D/texture_unlit.frag");
    cpl_create_shader(&_cpl_shaders[TEXTURE_2D_LIT],
                      "shaders/vert/2D/texture.vert",
                      "shaders/frag/2D/texture_lit.frag");
#else
    cpl_create_shader(&_cpl_shaders[SHAPE_2D_UNLIT],
                      "/shaders/vert/2D/shape_w.vert",
                      "/shaders/frag/2D/shape_unlit_w.frag");
    cpl_create_shader(&_cpl_shaders[SHAPE_2D_LIT],
                      "/shaders/vert/2D/shape_w.vert",
                      "/shaders/frag/2D/shape_lit_w.frag");
    cpl_create_shader(&_cpl_shaders[TEXT], "/shaders/vert/2D/text_w.vert",
                      "/shaders/frag/2D/text_w.frag");
    cpl_create_shader(&_cpl_shaders[TEXTURE_2D_UNLIT],
                      "/shaders/vert/2D/texture_w.vert",
                      "/shaders/frag/2D/texture_unlit_w.frag");
    cpl_create_shader(&_cpl_shaders[TEXTURE_2D_LIT],
                      "/shaders/vert/2D/texture_w.vert",
                      "/shaders/frag/2D/texture_lit_w.frag");
#endif
}

void cpl_init_window(u32 width, u32 height, c8 *title, u32 version) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, (i32)version / 10);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, (i32)version % 10);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef OPENGL_DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
#else
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 0);
#endif

#ifdef __EMSCRIPTEN__
    double browser_w, browser_h;
    emscripten_get_element_css_size("#canvas", &browser_w, &browser_h);
    width = (i32)browser_w;
    height = (i32)browser_h;
    if (width <= 0)
        width = emscripten_run_script_int("window.innerWidth");
    if (height <= 0)
        height = emscripten_run_script_int("window.innerHeight");
#endif

    _cpl_screen_width = width;
    _cpl_screen_height = height;

    _cpl_window =
        glfwCreateWindow((i32)width, (i32)height, title, NULLPTR, NULLPTR);
    if (!_cpl_window) {
        cpl_log(LOG_ERR, "[CPL] [ERROR] Failed to create window");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(_cpl_window);
    glfwSetFramebufferSizeCallback(_cpl_window, _cpl_framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)(glfwGetProcAddress))) {
        cpl_log(LOG_ERR, "[CPL] [ERROR] Failed to initialize GLAD");
        exit(-1);
    }

    cpl_enable_opengl_debug();

    _cpl_cam_2D = (cam_2D){{0.0f, 0.0f}, 1.0f, 0.0f};
    mat4f_ortho(&_cpl_projection_2D, 0.0f, (f32)_cpl_screen_width,
                (f32)_cpl_screen_height, 0.0f, -1.0f, 1.0f);

    _cpl_init_shaders();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    _cpl_renderer = (GLubyte *)glGetString(GL_RENDERER);
    _cpl_vendor = (GLubyte *)glGetString(GL_VENDOR);
    _cpl_version = (GLubyte *)glGetString(GL_VERSION);
}

b8 cpl_window_should_close() { return glfwWindowShouldClose(_cpl_window); }

void cpl_destroy_window() { glfwSetWindowShouldClose(_cpl_window, 1); }

void cpl_close_window() { glfwTerminate(); }

u32 cpl_get_screen_width() { return _cpl_screen_width; }
u32 cpl_get_screen_height() { return _cpl_screen_height; }
vec2f cpl_get_screen_size() {
    return VEC2F(_cpl_screen_width, _cpl_screen_height);
}

void cpl_enable_vsync(b8 enabled) { glfwSwapInterval(enabled); }

void cpl_update() {
    cpl_calc_fps();
    cpl_calc_dt();
    cpl_update_input();
}

void cpl_end_frame() {
    glfwSwapBuffers(_cpl_window);
    glfwPollEvents();
}
#endif

// }}}

// {{{ Drawing

void cpl_clear_background(vec4f color);
void cpl_begin_draw(cpl_draw_mode draw_mode, b8 mode_2D);
void cpl_draw_rect(vec2f pos, vec2f size, vec4f color, f32 rot);
void cpl_draw_triangle(vec2f pos, vec2f size, vec4f color, f32 rot);
void cpl_draw_circle(vec2f pos, f32 radius, vec4f color);
void cpl_draw_line(vec2f start, vec2f end, f32 thickness, vec4f color);
void cpl_draw_text(font *font, c8 *text, vec2f pos, f32 scale, vec4f color);
void cpl_draw_text_shadow(font *font, c8 *text, vec2f pos, f32 scale,
                          vec4f color, vec2f shadow_off, vec4f shadow_color);
void cpl_draw_texture2D(texture *tex, vec2f pos, vec2f size, vec4f color,
                        f32 rot);
void _cpl_reset_shader();
void cpl_display_details(font *font);

#ifdef CPL_IMPLEMENTATION
void cpl_clear_background(vec4f color) {
    glClearColor(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f,
                 color.a / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void cpl_begin_draw(cpl_draw_mode draw_mode, b8 mode_2D) {
    _cpl_cur_draw_mode = draw_mode;
    cpl_use_shader(&_cpl_shaders[draw_mode]);

    mat4f view_projection_2D;
    if (mode_2D) {
        mat4f *view = cpl_cam_2D_get_view_mat(&_cpl_cam_2D);
        mat4f_mul(&_cpl_projection_2D, view, &view_projection_2D);
        free(view);
    }
    cpl_shader_set_mat4f(&_cpl_shaders[draw_mode], "projection",
                         mode_2D ? view_projection_2D : _cpl_projection_2D);
}

void cpl_draw_rect(vec2f pos, vec2f size, vec4f color, f32 rot) {
    rect r;
    cpl_create_rect(&r, pos, size, color, rot);
    cpl_draw_rect_raw(&_cpl_shaders[_cpl_cur_draw_mode], &r);
    cpl_destroy_rect(&r);
}

void cpl_draw_triangle(vec2f pos, vec2f size, vec4f color, f32 rot) {
    triangle t;
    cpl_create_triangle(&t, pos, size, color, rot);
    cpl_draw_triangle_raw(&_cpl_shaders[_cpl_cur_draw_mode], &t);
    cpl_destroy_triangle(&t);
}

void cpl_draw_circle(vec2f pos, f32 radius, vec4f color) {
    circle c;
    cpl_create_circle(&c, pos, radius, color);
    cpl_draw_circle_raw(&_cpl_shaders[_cpl_cur_draw_mode], &c);
    cpl_destroy_circle(&c);
}

void cpl_draw_line(vec2f start, vec2f end, f32 thickness, vec4f color) {
    line l;
    cpl_create_line(&l, start, end, color);
    glLineWidth(thickness);
    cpl_draw_line_raw(&_cpl_shaders[_cpl_cur_draw_mode], &l);
    glLineWidth(1.0f);
    cpl_destroy_line(&l);
}

void cpl_draw_text(font *font, c8 *text, vec2f pos, f32 scale, vec4f color) {
    cpl_draw_text_raw(&_cpl_shaders[_cpl_cur_draw_mode], font, text, pos, scale,
                      color);
}

void cpl_draw_text_shadow(font *font, c8 *text, vec2f pos, f32 scale,
                          vec4f color, vec2f shadow_off, vec4f shadow_color) {
    cpl_draw_text_raw(&_cpl_shaders[_cpl_cur_draw_mode], font, text,
                      VEC2F(pos.x + shadow_off.x, pos.y + shadow_off.y), scale,
                      shadow_color);
    cpl_draw_text_raw(&_cpl_shaders[_cpl_cur_draw_mode], font, text, pos, scale,
                      color);
}

void cpl_draw_texture2D(texture *tex, vec2f pos, vec2f size, vec4f color,
                        f32 rot) {
    texture2D t;
    cpl_create_texture2D(&t, pos, size, rot, color, tex);
    cpl_draw_texture2D_raw(&_cpl_shaders[_cpl_cur_draw_mode], &t);
    cpl_destroy_texture2D(&t);
}

void cpl_reset_shader() { cpl_use_shader(&_cpl_shaders[_cpl_cur_draw_mode]); }

void cpl_display_details(font *font) {
    cpl_begin_draw(TEXT, false);

    c8 version_str[50];
    c8 renderer_str[50];
    c8 vendor_str[50];
    c8 fps[15];
    c8 stack_used[50];
    c8 heap_total[50];
    c8 heap_used[50];
    c8 heap_free[50];

    snprintf(version_str, 50, "OpenGL version: %s", _cpl_version);
    snprintf(renderer_str, 50, "Renderer: %s", _cpl_renderer);
    snprintf(vendor_str, 50, "Vendor: %s", _cpl_vendor);
    snprintf(fps, 15, "FPS: %d", cpl_get_fps());
    snprintf(stack_used, 50, "Stack used: %.3f / %.3f MB (%f%%)",
             MB((f32)cpl_get_stack_used()), MB((f32)cpl_get_stack_size()),
             (f32)cpl_get_stack_used() / (f32)cpl_get_stack_size());
    snprintf(heap_total, 50, "Heap size: %d MB",
             (i32)MB((f32)cpl_get_heap_size()));
    snprintf(heap_used, 50, "Heap used: %d MB",
             (i32)MB((f32)cpl_get_heap_used()));
    snprintf(heap_free, 50, "Heap free: %d MB",
             (i32)MB((f32)cpl_get_heap_free()));

    cpl_draw_text(font, version_str, VEC2F(10.0f, 10.0f), 0.5f, WHITE);
    cpl_draw_text(font, renderer_str, VEC2F(10.0f, 40.0f), 0.5f, WHITE);
    cpl_draw_text(font, vendor_str, VEC2F(10.0f, 70.0f), 0.5f, WHITE);
    cpl_draw_text(font, fps, VEC2F(10.0f, 100.0f), 0.5f, WHITE);
    cpl_draw_text(font, stack_used, VEC2F(10.0f, 130.0f), 0.5f, WHITE);
    cpl_draw_text(font, heap_total, VEC2F(10.0f, 160.0f), 0.5f, WHITE);
    cpl_draw_text(font, heap_used, VEC2F(10.0f, 190.0f), 0.5f, WHITE);
    cpl_draw_text(font, heap_free, VEC2F(10.0f, 220.0f), 0.5f, WHITE);
}
#endif

// }}}

// {{{ Lighting 2D

typedef struct {
    vec2f pos;
    f32 radius;
    f32 intensity;
    vec4f color;
} point_light_2D;
typedef struct {
    f32 intensity;
    vec4f color;
} global_light_2D;

void cpl_set_ambient_light_2D(f32 strength);
void cpl_set_global_light_2D(global_light_2D *l);
void cpl_add_point_lights_2D(point_light_2D *ls, u32 size);

#ifdef CPL_IMPLEMENTATION
void cpl_set_ambient_light_2D(f32 strength) {
    shader *ss = &_cpl_shaders[SHAPE_2D_LIT];
    cpl_use_shader(ss);
    cpl_shader_set_f32(ss, "ambient", strength);

    shader *ts = &_cpl_shaders[TEXTURE_2D_LIT];
    cpl_use_shader(ts);
    cpl_shader_set_f32(ts, "ambient", strength);

    cpl_reset_shader();
}
void cpl_set_global_light_2D(global_light_2D *l) {
    shader *ss = &_cpl_shaders[SHAPE_2D_LIT];
    cpl_use_shader(ss);
    cpl_shader_set_f32(ss, "g_light.intensity", l->intensity);
    cpl_shader_set_rgba(ss, "g_light.color", l->color);

    shader *ts = &_cpl_shaders[TEXTURE_2D_LIT];
    cpl_use_shader(ts);
    cpl_shader_set_f32(ts, "g_light.intensity", l->intensity);
    cpl_shader_set_rgba(ts, "g_light.color", l->color);

    cpl_reset_shader();
}

void cpl_add_point_lights_2D(point_light_2D *ls, u32 size) {
    shader *ss = &_cpl_shaders[SHAPE_2D_LIT];
    cpl_use_shader(ss);

    cpl_shader_set_i32(ss, "point_lights_cnt", (i32)size);
    for (u32 i = 0; i < size; i++) {
        c8 pos[50];
        snprintf(pos, 50, "point_lights[%d].pos", i);
        c8 radius[50];
        snprintf(radius, 50, "point_lights[%d].r", i);
        c8 intensity[50];
        snprintf(intensity, 50, "point_lights[%d].intensity", i);
        c8 color[50];
        snprintf(color, 50, "point_lights[%d].color", i);

        cpl_shader_set_vec2f(ss, pos, ls[i].pos);
        cpl_shader_set_f32(ss, radius, ls[i].radius);
        cpl_shader_set_f32(ss, intensity, ls[i].intensity);
        cpl_shader_set_rgba(ss, color, ls[i].color);
    }

    shader *ts = &_cpl_shaders[TEXTURE_2D_LIT];
    cpl_use_shader(ts);

    cpl_shader_set_i32(ts, "point_lights_cnt", (i32)size);
    for (u32 i = 0; i < size; i++) {
        c8 pos[50];
        snprintf(pos, 50, "point_lights[%d].pos", i);
        c8 radius[50];
        snprintf(radius, 50, "point_lights[%d].r", i);
        c8 intensity[50];
        snprintf(intensity, 50, "point_lights[%d].intensity", i);
        c8 color[50];
        snprintf(color, 50, "point_lights[%d].color", i);

        cpl_shader_set_vec2f(ts, pos, ls[i].pos);
        cpl_shader_set_f32(ts, radius, ls[i].radius);
        cpl_shader_set_f32(ts, intensity, ls[i].intensity);
        cpl_shader_set_rgba(ts, color, ls[i].color);
    }

    cpl_reset_shader();
}
#endif

// }}}

// {{{ Tilemap

#define CPL_TILEMAP_ABS_UV(m, vx, vy)                                          \
    VEC2F((vx) / (m).tex.size.x, (vy) / (m).tex.size.y)

typedef struct {
    f32 vertices[30];
    u32 vbo;
} tile_batch;

VEC_DEF(tile_batch, cpl_tile_batches)
VEC_DEF(vec2f, cpl_tiles)

typedef struct {
    cpl_tiles tiles;
    cpl_tile_batches batches;
    vec2f size;
    texture tex;
    u32 vao;
} tilemap;

void cpl_create_tilemap(tilemap *m);
void cpl_destroy_tilemap(tilemap *m);
void cpl_tilemap_load_texture(tilemap *m, c8 *path, texture_filtering filter);
void cpl_tilemap_set_tile_size(tilemap *m, vec2f size);
void cpl_tilemap_begin_editing(tilemap *m);
void cpl_tilemap_add_tile(tilemap *m, vec2f pos, vec2f size, vec2f uv);
void cpl_tilemap_delete_tile(tilemap *m, vec2f pos);
b8 cpl_tilemap_tile_exists(tilemap *m, vec2f pos);
void cpl_tilemap_draw(tilemap *m);

#ifdef CPL_IMPLEMENTATION
void cpl_create_tilemap(tilemap *m) {
    cpl_tiles_reserve(&m->tiles, 10);
    cpl_tile_batches_reserve(&m->batches, 10);
    glGenVertexArrays(1, &m->vao);
}

void cpl_destroy_tilemap(tilemap *m) {
    cpl_tiles_destroy(&m->tiles);
    cpl_tile_batches_destroy(&m->batches);
    if (m->vao != 0 && glIsVertexArray(m->vao)) {
        glDeleteVertexArrays(1, &m->vao);
        m->vao = 0;
    }
}

void cpl_tilemap_load_texture(tilemap *m, c8 *path, texture_filtering filter) {
    cpl_load_texture(&m->tex, path, filter);
}

void cpl_tilemap_set_tile_size(tilemap *m, vec2f size) { m->size = size; }

void cpl_tilemap_begin_editing(tilemap *m) {
    cpl_tile_batches_clear(&m->batches);
}

void cpl_tilemap_add_tile(tilemap *m, vec2f pos, vec2f size, vec2f uv) {
    f32 tw = m->size.x / m->tex.size.x;
    f32 th = m->size.y / m->tex.size.y;

    f32 quad[30] = {
        pos.x,          pos.y,          0, uv.x,      uv.y + th,
        pos.x + size.x, pos.y,          0, uv.x + tw, uv.y + th,
        pos.x + size.x, pos.y + size.y, 0, uv.x + tw, uv.y,
        pos.x,          pos.y,          0, uv.x,      uv.y + th,
        pos.x + size.x, pos.y + size.y, 0, uv.x + tw, uv.y,
        pos.x,          pos.y + size.y, 0, uv.x,      uv.y,
    };

    tile_batch batch;
    glGenBuffers(1, &batch.vbo);
    memcpy(batch.vertices, quad, sizeof(quad));
    cpl_tile_batches_push_back(&m->batches, batch);

    cpl_tiles_push_back(&m->tiles, pos);
}

void cpl_tilemap_delete_tile(tilemap *m, vec2f pos) {
    for (int i = 0; i < m->tiles.size; i++) {
        vec2f *tile = cpl_tiles_at(&m->tiles, i);
        if (tile->x == pos.x && tile->y == pos.y) {
            cpl_tiles_delete(&m->tiles, i);
            break;
        }
    }
    for (int i = 0; i < m->batches.size; i++) {
        if (cpl_tile_batches_at(&m->batches, i)->vertices[0] == pos.x &&
            cpl_tile_batches_at(&m->batches, i)->vertices[1] == pos.y) {
            cpl_tile_batches_delete(&m->batches, i);
            break;
        }
    }
}

b8 cpl_tilemap_tile_exists(tilemap *m, vec2f pos) {
    FOREACH_VEC(vec2f, cpl_tiles, p, &m->tiles) {
        if (p->x == pos.x && p->y == pos.y) {
            return true;
        }
    }
    return false;
}

void cpl_tilemap_draw(tilemap *m) {
    mat4f transform;
    mat4f_identity(&transform);

    cpl_shader_set_mat4f(&_cpl_shaders[_cpl_cur_draw_mode], "transform",
                         transform);
    cpl_shader_set_rgba(&_cpl_shaders[_cpl_cur_draw_mode], "input_color",
                        WHITE);

    glBindVertexArray(m->vao);

    FOREACH_VEC(tile_batch, cpl_tile_batches, batch, &m->batches) {
        glBindBuffer(GL_ARRAY_BUFFER, batch->vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(batch->vertices), batch->vertices,
                     GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                              (void *)NULLPTR);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(f32),
                              (void *)(3 * sizeof(f32)));
        glEnableVertexAttribArray(1);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m->tex.id);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glBindVertexArray(0);
}
#endif

// }}}

// {{{ Particle System

#define UNLIMITED_PARTICLES 0

#define PARTICLE(pos, size, dir, color, life_time, rot, tex)                   \
    (particle) { pos, size, dir, color, 0, life_time, rot, tex, true }

typedef struct {
    vec2f pos;
    vec2f size;
    vec2f dir;
    vec4f color;
    f32 cur_life_time;
    f32 life_time;
    f32 rot;
    texture *tex;
    b8 active;
} particle;

VEC_DEF(particle, vec_particle)

typedef struct {
    vec2f pos;
    u32 max_particles;

    vec_particle particles;
} particle_system;

void cpl_create_particle_system(particle_system *ps, vec2f pos,
                                u32 max_particles);
void cpl_destroy_particle_system(particle_system *ps);
void cpl_update_particle_system(particle_system *ps);
void cpl_draw_particles(particle_system *ps);
void cpl_add_particle(particle_system *ps, particle p);

#ifdef CPL_IMPLEMENTATION
void cpl_create_particle_system(particle_system *ps, vec2f pos,
                                u32 max_particles) {
    ps->pos = pos;
    ps->max_particles = max_particles;
    vec_particle_reserve(&ps->particles,
                         max_particles >= 10 ||
                                 max_particles == UNLIMITED_PARTICLES
                             ? 10
                             : max_particles);
}

void cpl_destroy_particle_system(particle_system *ps) {
    vec_particle_destroy(&ps->particles);
}

void cpl_update_particle_system(particle_system *ps) {
    FOREACH_VEC(particle, vec_particle, p, &ps->particles) {
        p->cur_life_time += cpl_get_dt();
        p->pos = vec2f_add(
            &VEC2F(p->dir.x * cpl_get_dt(), p->dir.y * cpl_get_dt()), &p->pos);
        if (p->cur_life_time >= p->life_time) {
            p->active = false;
        }
    }

    VEC_ERASE_IF(&ps->particles, !it.active);
}

void cpl_draw_particles(particle_system *ps) {
    FOREACH_VEC(particle, vec_particle, p, &ps->particles) {
        cpl_draw_texture2D(p->tex, p->pos, p->size, p->color, p->rot);
    }
}

void cpl_add_particle(particle_system *ps, particle p) {
    if (ps->particles.size < ps->max_particles || ps->max_particles == 0) {
        vec_particle_push_back(&ps->particles, p);
    }
}
#endif

// }}}

// {{{ Shadow 2D

#define MAX_RECT_SHADOWS 1024

typedef struct {
    vec2f pos;
    vec2f size;
} rect_shadow;

rect_shadow _cpl_rect_shadows[MAX_RECT_SHADOWS];
u32 _cpl_rect_shadow_count = 0;

void cpl_begin_shadow_cast_2D();
void cpl_end_shadow_cast_2D(f32 ambient, f32 shadow_strength,
                            color shadow_color);
void cpl_submit_rect_shadow(vec2f pos, vec2f size);
void cpl_draw_triangle_shadow(vec2f a, vec2f b, vec2f c);
void cpl_draw_rect_shadow(vec2f pos, vec2f size, point_light_2D *lights, u32 n,
                          f32 far);
void cpl_draw_shadows(point_light_2D *lights, u32 light_count, f32 far,
                      f32 shadow_strength);

#ifdef CPL_IMPLEMENTATION
void cpl_begin_shadow_cast_2D() {
    glEnable(GL_STENCIL_TEST);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    cpl_begin_draw(SHAPE_2D_UNLIT, true);
}

void cpl_end_shadow_cast_2D(f32 ambient, f32 shadow_strength,
                            color shadow_color) {
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    f32 shadow_alpha = CPM_CLAMP(shadow_strength - ambient, 0, 1);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    cpl_begin_draw(SHAPE_2D_UNLIT, false);
    cpl_draw_rect(VEC2F_INIT(0), cpl_get_screen_size(),
                  RGBA(shadow_color.r, shadow_color.g, shadow_color.b,
                       255 * shadow_alpha),
                  0);
    glDisable(GL_STENCIL_TEST);
}

void cpl_submit_rect_shadow(vec2f pos, vec2f size) {
    if (_cpl_rect_shadow_count < MAX_RECT_SHADOWS) {
        _cpl_rect_shadows[_cpl_rect_shadow_count++] = (rect_shadow){pos, size};
    }
}

// TODO make ts somehow work

void cpl_draw_triangle_shadow(vec2f a, vec2f b, vec2f c) {
    f32 vertices[9] = {
        a.x, a.y, 0.0f, b.x, b.y, 0.0f, c.x, c.y, 1.0f,
    };
    u32 vao;
    u32 vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULL);
    glEnableVertexAttribArray(0);
    mat4f transform;
    mat4f_identity(&transform);
    cpl_shader_set_mat4f(&_cpl_shaders[SHAPE_2D_UNLIT], "transform", transform);
    cpl_shader_set_rgba(&_cpl_shaders[SHAPE_2D_UNLIT], "input_color",
                        RGBA(0.0f, 0.0f, 0.0f, 1.0f));
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void cpl_draw_rect_shadow(vec2f pos, vec2f size, point_light_2D *lights, u32 n,
                          f32 far) {
    vec2f corners[4] = {
        {pos.x, pos.y},
        {pos.x + size.x, pos.y},
        {pos.x + size.x, pos.y + size.y},
        {pos.x, pos.y + size.y},
    };
    vec2f normals[4] = {
        {0, -1},
        {1, 0},
        {0, 1},
        {-1, 0},
    };
    for (u32 l = 0; l < n; l++) {
        for (u32 i = 0; i < 4; i++) {
            u32 next = (i + 1) % 4;
            vec2f a = corners[i];
            vec2f b = corners[next];
            vec2f mid = {(a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f};
            vec2f to_edge = {mid.x - lights[l].pos.x, mid.y - lights[l].pos.y};
            f32 d = (normals[i].x * to_edge.x) + (normals[i].y * to_edge.y);
            if (d <= 0) {
                continue;
            }
            vec2f dir_a = {a.x - lights[l].pos.x, a.y - lights[l].pos.y};
            f32 len_a = cpm_sqrt((dir_a.x * dir_a.x) + (dir_a.y * dir_a.y));
            dir_a.x /= len_a;
            dir_a.y /= len_a;
            vec2f dir_b = {b.x - lights[l].pos.x, b.y - lights[l].pos.y};
            f32 len_b = cpm_sqrt((dir_b.x * dir_b.x) + (dir_b.y * dir_b.y));
            dir_b.x /= len_b;
            dir_b.y /= len_b;
            vec2f a2 = {a.x + (dir_a.x * far), a.y + (dir_a.y * far)};
            vec2f b2 = {b.x + (dir_b.x * far), b.y + (dir_b.y * far)};
            cpl_draw_triangle_shadow(a, b, b2);
            cpl_draw_triangle_shadow(a, b2, a2);
        }
    }
}

void cpl_draw_shadows(point_light_2D *lights, u32 light_count, f32 far,
                      f32 shadow_strength) {
    glEnable(GL_STENCIL_TEST);

    glClear(GL_STENCIL_BUFFER_BIT);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    for (u32 l = 0; l < light_count; l++) {
        for (u32 i = 0; i < _cpl_rect_shadow_count; i++) {
            cpl_draw_rect_shadow(_cpl_rect_shadows[i].pos,
                                 _cpl_rect_shadows[i].size, &lights[l], 1, far);
        }
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

    cpl_begin_draw(SHAPE_2D_UNLIT, false);
    cpl_draw_rect(VEC2F_INIT(0), cpl_get_screen_size(),
                  RGBA(0, 0, 0, (u8)(255 * shadow_strength)), 0);

    glDisable(GL_STENCIL_TEST);
    _cpl_rect_shadow_count = 0;
}
#endif

// }}}
