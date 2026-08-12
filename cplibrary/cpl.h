#pragma once

#ifdef __linux__
#define _GNU_SOURCE
#endif

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#ifdef __linux__
#include <malloc.h>
#endif

#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef CPL_IMPL
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define MINIAUDIO_IMPLEMENTATION
#define ENET_IMPLEMENTATION
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

#include "stb_image.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include "stb_image_write.h"
#include <enet.h>
#include <miniaudio.h>

#include "../cpstd/arena.h"
#include "../cpstd/mathplus.h"
#include "../cpstd/vector.h"

// Available macros
// #define CPL_IMPL
// #define CPL_INCLUDE_NETWORKING
// #define CPL_INCLUDE_DEBUG

#pragma region Key Inputs

typedef enum {
    MOUSE_BUTTON_1 = 0,
    MOUSE_BUTTON_2 = 1,
    MOUSE_BUTTON_3 = 2,
    MOUSE_BUTTON_4 = 3,
    MOUSE_BUTTON_5 = 4,
    MOUSE_BUTTON_6 = 5,
    MOUSE_BUTTON_7 = 6,
    MOUSE_BUTTON_8 = 7,
    MOUSE_BUTTON_LAST = MOUSE_BUTTON_8,
    MOUSE_BUTTON_LEFT = MOUSE_BUTTON_1,
    MOUSE_BUTTON_RIGHT = MOUSE_BUTTON_2,
    MOUSE_BUTTON_MIDDLE = MOUSE_BUTTON_3,
} mouse_button_t;

typedef enum {
    KEY_SPACE = 32,
    KEY_APOSTROPHE = 39,
    KEY_COMMA = 44,
    KEY_MINUS,
    KEY_PERIOD,
    KEY_SLASH,
    KEY_DIGIT_0,
    KEY_DIGIT_1,
    KEY_DIGIT_2,
    KEY_DIGIT_3,
    KEY_DIGIT_4,
    KEY_DIGIT_5,
    KEY_DIGIT_6,
    KEY_DIGIT_7,
    KEY_DIGIT_8,
    KEY_DIGIT_9,
    KEY_SEMICOLON = 59,
    KEY_EQUAL = 61,
    KEY_LETTER_A = 65,
    KEY_LETTER_B,
    KEY_LETTER_C,
    KEY_LETTER_D,
    KEY_LETTER_E,
    KEY_LETTER_F,
    KEY_LETTER_G,
    KEY_LETTER_H,
    KEY_LETTER_I,
    KEY_LETTER_J,
    KEY_LETTER_K,
    KEY_LETTER_L,
    KEY_LETTER_M,
    KEY_LETTER_N,
    KEY_LETTER_O,
    KEY_LETTER_P,
    KEY_LETTER_Q,
    KEY_LETTER_R,
    KEY_LETTER_S,
    KEY_LETTER_T,
    KEY_LETTER_U,
    KEY_LETTER_V,
    KEY_LETTER_W,
    KEY_LETTER_X,
    KEY_LETTER_Y,
    KEY_LETTER_Z,
    KEY_LEFT_BRACKET,
    KEY_BACKSLASH,
    KEY_RIGHT_BRACKET,
    KEY_GRAVE_ACCENT = 96,
    KEY_WORLD_1 = 161,
    KEY_WORLD_2,
    KEY_ESCAPE = 256,
    KEY_ENTER,
    KEY_TAB,
    KEY_BACKSPACE,
    KEY_INSERT,
    KEY_DELETE,
    KEY_RIGHT_ARROW,
    KEY_LEFT_ARROW,
    KEY_DOWN_ARROW,
    KEY_UP_ARROW,
    KEY_PAGE_UP,
    KEY_PAGE_DOWN,
    KEY_HOME,
    KEY_END,
    KEY_CAPS_LOCK = 280,
    KEY_SCROLL_LOCK,
    KEY_NUM_LOCK,
    KEY_PRINT_SCREEN,
    KEY_PAUSE,
    KEY_F1 = 290,
    KEY_F2,
    KEY_F3,
    KEY_F4,
    KEY_F5,
    KEY_F6,
    KEY_F7,
    KEY_F8,
    KEY_F9,
    KEY_F10,
    KEY_F11,
    KEY_F12,
    KEY_F13,
    KEY_F14,
    KEY_F15,
    KEY_F16,
    KEY_F17,
    KEY_F18,
    KEY_F19,
    KEY_F20,
    KEY_F21,
    KEY_F22,
    KEY_F23,
    KEY_F24,
    KEY_F25,
    KEY_KP_0 = 320,
    KEY_KP_1,
    KEY_KP_2,
    KEY_KP_3,
    KEY_KP_4,
    KEY_KP_5,
    KEY_KP_6,
    KEY_KP_7,
    KEY_KP_8,
    KEY_KP_9,
    KEY_KP_DECIMAL,
    KEY_KP_DIVIDE,
    KEY_KP_MULTIPLY,
    KEY_KP_SUBTRACT,
    KEY_KP_ADD,
    KEY_KP_ENTER,
    KEY_KP_EQUAL,
    KEY_LEFT_SHIFT = 340,
    KEY_LEFT_CONTROL,
    KEY_LEFT_ALT,
    KEY_LEFT_SUPER,
    KEY_RIGHT_SHIFT,
    KEY_RIGHT_CONTROL,
    KEY_RIGHT_ALT,
    KEY_RIGHT_SUPER,
    KEY_MENU,
    KEY_LAST = GLFW_KEY_MENU
} key_button_t;

#pragma endregion

#pragma region OpenGL Versions

typedef enum {
    OPENGL_VER_1_0 = 10,
    OPENGL_VER_2_0 = 20,
    OPENGL_VER_3_0 = 30,
    OPENGL_VER_3_3 = 33,
    OPENGL_VER_4_0 = 40,
    OPENGL_VER_4_1 = 41,
    OPENGL_VER_4_2 = 42,
    OPENGL_VER_4_3 = 43,
    OPENGL_VER_4_4 = 44,
    OPENGL_VER_4_5 = 45,
    OPENGL_VER_4_6 = 46
} opengl_version_t;

#pragma endregion

#pragma region Colors

typedef struct { 
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} color_t;

#define RGB(r, g, b) (color_t){r, g, b, 255}
#define RGBA(r, g, b, a) (color_t) { r, g, b, a }

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

#pragma endregion

#ifdef __cplusplus
namespace cpl {
extern "C" {
#endif

#pragma region Logging

typedef enum { 
    LOG_INFO = 0, 
    LOG_WARN, 
    LOG_ERR, 
    LOG_NONE 
} log_level_t;

void cpl_log(log_level_t level, char *msg, ...);

#ifdef CPL_IMPL

void cpl_log(log_level_t level, char *msg, ...) {
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
    while (*msg) {
        if ((*msg) == '%') {
            msg++;
            switch ((*msg)) {
            case 'c': {
                int c = va_arg(args, int);
                putchar(c);
                break;
            }
            case 'i': {
                int i = va_arg(args, int);
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
                char *s = va_arg(args, char *);
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

#pragma endregion

#pragma region Screenshot

void screenshot_take(char *path, vec2f screen);

#ifdef CPL_IMPL

void screenshot_take(char *path, vec2f screen) {
    static int screenshots_taken = 0;
    int w = (int)screen.x;
    int h = (int)screen.y;
    int s = w * 3;
    uint8_t pixels[w * h * 3];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < s; x++) {
            uint8_t tmp = pixels[(y * s) + x];
            pixels[(y * s) + x] = pixels[((h - 1 - y) * s) + x];
            pixels[((h - 1 - y) * s) + x] = tmp;
        }
    }
    char final[100];
    snprintf(final, sizeof(final), "%sscreenshot%d.png", path, screenshots_taken);
    stbi_write_png(final, w, h, 3, pixels, s);
    screenshots_taken++;
}

#endif

#pragma endregion

#pragma region Profiler

#ifdef __linux__

unsigned int profiler_get_heap_size();
unsigned int profiler_get_heap_used();
unsigned int profiler_get_heap_free();
unsigned int profiler_get_stack_size();
unsigned int profiler_get_stack_used();

#ifdef CPL_IMPL

unsigned int profiler_get_heap_size() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.arena;
#else
    return 0;
#endif
}
unsigned int profiler_get_heap_used() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.uordblks;
#else
    return 0;
#endif
}
unsigned int profiler_get_heap_free() {
#ifndef __EMSCRIPTEN__
    struct mallinfo2 mi = mallinfo2();
    return mi.fordblks;
#else
    return 0;
#endif
}
unsigned int profiler_get_stack_size() {
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
unsigned int profiler_get_stack_used() {
#ifndef __EMSCRIPTEN__
    pthread_attr_t attr;
    pthread_getattr_np(pthread_self(), &attr);
    void *base = NULL;
    size_t size = 0;
    pthread_attr_getstack(&attr, &base, &size);
    pthread_attr_destroy(&attr);
    char marker;
    void *cur = &marker;
    return (unsigned int)(base + size - cur);
#else
    return 0;
#endif
}

#endif

#endif

#pragma endregion

#ifdef CPL_INCLUDE_DEBUG

#pragma region OpenGL Debug

GLenum _opengl_debug_check_error(char *path, unsigned int line);
void opengl_debug_check_error();
void APIENTRY _opengl_debug_out(GLenum src, GLenum type, unsigned int id, GLenum severity, GLsizei len, const char *msg, const void *usr_prog);
void opengl_debug_enable();

#ifdef CPL_IMPL

GLenum _opengl_debug_check_error(char *path, unsigned int line) {
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR) {
        char *error;
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
void opengl_debug_check_error() {
    _opengl_debug_check_error(__FILE__, __LINE__);
}
void APIENTRY _opengl_debug_out(GLenum src, GLenum type, unsigned int id, GLenum severity, GLsizei len, const char *msg, const void *usr_prog) {
    (void)len;
    (void)usr_prog;

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
void opengl_debug_enable() {
    int flags = 0;
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
        glDebugMessageCallback(_opengl_debug_out, NULL);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL,
                              GL_TRUE);
    }
}

#endif

#pragma endregion

#endif

#pragma region Shader

typedef struct {
    unsigned int id;
} shader_t;

bool _shader_check_compile_errors(unsigned int shader, char *type);
char *_shader_read_file(char *path);
void shader_create(shader_t *s, char *vert_path, char *frag_path);
void shader_use(shader_t *s);
void shader_set_bool(shader_t *s, char *name, bool val);
void shader_set_int(shader_t *s, char *name, int val);
void shader_set_float(shader_t *s, char *name, float val);
void shader_set_color(shader_t *s, char *name, color_t c);
void shader_set_mat4f(shader_t *s, char *name, mat4f *mat);
void shader_set_vec2f(shader_t *s, char *name, vec2f v);
void shader_set_vec3f(shader_t *s, char *name, vec3f v);

#ifdef CPL_IMPL

bool _shader_check_compile_errors(unsigned int shader, char *type) {
    int success = 0;
    char info_cpl_log[1024];

    if (strcmp(type, "PROGRAM") == 0) {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, info_cpl_log);
            cpl_log(LOG_ERR, "[CPL] [ERROR] Program linking error:\n%s\n", info_cpl_log);
            return false;
        }
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, info_cpl_log);
            cpl_log(LOG_ERR, "[CPL] [ERROR] Shader compilation error: %s\n%s\n", type, info_cpl_log);
            return false;
        }
    }
    return true;
}
char *_shader_read_file(char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    unsigned int size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    unsigned int read = fread(buffer, 1, size, f);
    fclose(f);

    if (read != size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    return buffer;
}
void shader_create(shader_t *s, char *vert_path, char *frag_path) {
    char *vert_code = _shader_read_file(vert_path);
    char *frag_code = _shader_read_file(frag_path);

    unsigned int vert = 0;
    unsigned int frag = 0;
    vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, (const GLchar *const *)&vert_code, NULL);
    glCompileShader(vert);
    _shader_check_compile_errors(vert, "VERTEX");
    frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, (const GLchar *const *)&frag_code, NULL);
    glCompileShader(frag);
    _shader_check_compile_errors(frag, "FRAGMENT");
    s->id = glCreateProgram();
    glAttachShader(s->id, vert);
    glAttachShader(s->id, frag);
    glLinkProgram(s->id);
    _shader_check_compile_errors(s->id, "PROGRAM");

    free(vert_code);
    free(frag_code);
    glDeleteShader(vert);
    glDeleteShader(frag);
}
void shader_use(shader_t *s) { 
    glUseProgram(s->id); 
}
void shader_set_bool(shader_t *s, char *name, bool val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}
void shader_set_int(shader_t *s, char *name, int val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}
void shader_set_float(shader_t *s, char *name, float val) {
    glUniform1f(glGetUniformLocation(s->id, name), val);
}
void shader_set_color(shader_t *s, char *name, color_t c) {
    glUniform4f(glGetUniformLocation(s->id, name), (float)c.r / 255.0f, (float)c.g / 255.0f, (float)c.b / 255.0f, (float)c.a / 255.0f);
}
void shader_set_mat4f(shader_t *s, char *name, mat4f *mat) {
    glUniformMatrix4fv(glGetUniformLocation(s->id, name), 1, GL_FALSE, (const GLfloat *)mat);
}
void shader_set_vec2f(shader_t *s, char *name, vec2f v) {
    glUniform2f(glGetUniformLocation(s->id, name), v.x, v.y);
}
void shader_set_vec3f(shader_t *s, char *name, vec3f v) {
    glUniform3f(glGetUniformLocation(s->id, name), v.x, v.y, v.z);
}

#endif

#pragma endregion

#pragma region Rectangle

typedef struct {
    vec2f pos;
    vec2f size;
    color_t color;
    float rot;
    unsigned int vbo, vao, ebo;
} rect_t;

void rect_create(rect_t *r, vec2f pos, vec2f size, color_t color, float rot);
void rect_destroy(rect_t *r);
void rect_draw_raw(shader_t *s, rect_t *r);

#ifdef CPL_IMPL

void rect_create(rect_t *r, vec2f pos, vec2f size, color_t color, float rot) {
    r->pos = pos;
    r->size = size;
    r->color = color;
    r->rot = rot;

    float vertices[12] = {
        size.x, 0.0f,   0.0f, 
        size.x, size.y, 0.0f, 
        0.0f,   size.y, 0.0f, 
        0.0f,   0.0f,   0.0f  
    };
    unsigned int indices[6] = {
        0, 1, 3, 
        1, 2, 3  
    };

    glGenVertexArrays(1, &r->vao);
    glGenBuffers(1, &r->vbo);
    glGenBuffers(1, &r->ebo);
    glBindVertexArray(r->vao);
    glBindBuffer(GL_ARRAY_BUFFER, r->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
void rect_destroy(rect_t *r) {
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
void rect_draw_raw(shader_t *s, rect_t *r) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, (vec3f){r->pos.x, r->pos.y, 0.0f});
    mat4f_translate(&transform, (vec3f){r->size.x * 0.5f, r->size.y * 0.5f, 0.0f});
    mat4f_rotate(&transform, math_rad(r->rot), (vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(&transform, (vec3f){-r->size.x * 0.5f, -r->size.y * 0.5f, 0.0f});

    shader_set_mat4f(s, "transform", &transform);
    shader_set_color(s, "input_color", r->color);

    glBindVertexArray(r->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

#endif

#pragma endregion

#pragma region Triangle

typedef struct {
    vec2f pos;
    vec2f size;
    color_t color;
    float rot;
    unsigned int vbo, vao;
} triangle_t;

void triangle_create(triangle_t *t, vec2f pos, vec2f size, color_t color, float rot);
void triangle_destroy(triangle_t *t);
void triangle_draw_raw(shader_t *s, triangle_t *t);

#ifdef CPL_IMPL

void triangle_create(triangle_t *t, vec2f pos, vec2f size, color_t color,
                     float rot) {
    t->pos = pos;
    t->size = size;
    t->color = color;
    t->rot = rot;

    float vertices[9] = {
        0.0f,          0.0f,   0.0f, 
        size.x,        0.0f,   0.0f, 
        size.x / 2.0f, size.y, 0.0f  
    };

    glGenVertexArrays(1, &t->vao);
    glGenBuffers(1, &t->vbo);
    glBindVertexArray(t->vao);
    glBindBuffer(GL_ARRAY_BUFFER, t->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
void triangle_destroy(triangle_t *t) {
    if (t->vao != 0 && glIsVertexArray(t->vao)) {
        glDeleteVertexArrays(1, &t->vao);
        t->vao = 0;
    }
    if (t->vbo != 0 && glIsBuffer(t->vbo)) {
        glDeleteBuffers(1, &t->vbo);
        t->vbo = 0;
    }
}
void triangle_draw_raw(shader_t *s, triangle_t *t) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, (vec3f){t->pos.x, t->pos.y, 0.0f});
    mat4f_translate(&transform, (vec3f){t->size.x * 0.5f, t->size.y * 0.5f, 0.0f});
    mat4f_rotate(&transform, math_rad(t->rot), (vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(&transform, (vec3f){-t->size.x * 0.5f, -t->size.y * 0.5f, 0.0f});

    shader_set_mat4f(s, "transform", &transform);
    shader_set_color(s, "input_color", t->color);

    glBindVertexArray(t->vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

#endif

#pragma endregion

#pragma region Circle

typedef struct {
    vec2f pos;
    float radius;
    color_t color;
    unsigned int vao, vbo;
    int vertex_cnt;
} circle_t;

void circle_create(circle_t *c, vec2f pos, float radius, color_t color);
void circle_destroy(circle_t *c);
void circle_draw_raw(shader_t *s, circle_t *c);

#ifdef CPL_IMPL

void circle_create(circle_t *c, vec2f pos, float radius, color_t color) {
    c->pos = pos;
    c->color = color;
    c->radius = radius;

    int segments = math_min(32, (int)ceilf(2.0f * MATH_PI * radius / 2.0f));
    size_t vertices_size = ((size_t)(segments + 1) * 3) + 3;
    float *vertices = malloc(vertices_size * sizeof(float));
    for (unsigned int i = 0; i < 3; i++) {
        vertices[i] = 0;
    }
    for (unsigned int i = 0; i <= segments; i++) {
        float theta = 2 * (float)MATH_PI / (float)segments * (float)i;
        float x = 0.0f + (radius * cosf(theta));
        float y = 0.0f + (radius * sinf(theta));
        vertices[(size_t)(i + 1) * 3] = x;
        vertices[((i + 1) * 3) + 1] = y;
        vertices[((i + 1) * 3) + 2] = 0.0f;
    }
    c->vertex_cnt = (int)(vertices_size / 3);
    glGenVertexArrays(1, &c->vao);
    glGenBuffers(1, &c->vbo);
    glBindVertexArray(c->vao);
    glBindBuffer(GL_ARRAY_BUFFER, c->vbo);
    glBufferData(GL_ARRAY_BUFFER, (int)(vertices_size * sizeof(float)), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    free(vertices);
}
void circle_destroy(circle_t *c) {
    if (c->vao != 0 && glIsVertexArray(c->vao)) {
        glDeleteVertexArrays(1, &c->vao);
        c->vao = 0;
    }
    if (c->vbo != 0 && glIsBuffer(c->vbo)) {
        glDeleteBuffers(1, &c->vbo);
        c->vbo = 0;
    }
}
void circle_draw_raw(shader_t *s, circle_t *c) {
    mat4f transform;
    mat4f_identity(&transform);
    mat4f_translate(&transform, (vec3f){c->pos.x, c->pos.y, 0.0f});

    shader_set_mat4f(s, "transform", &transform);
    shader_set_color(s, "input_color", c->color);

    glBindVertexArray(c->vao);
    glDrawArrays(GL_TRIANGLE_FAN, 0, c->vertex_cnt);
    glBindVertexArray(0);
}

#endif

#pragma endregion

#pragma region Line

typedef struct {
    vec2f start, end;
    color_t color;
    unsigned int vao, vbo;
} line_t;

void line_create(line_t *l, vec2f start, vec2f end, color_t color);
void line_destroy(line_t *l);
void line_draw_raw(shader_t *s, line_t *l);

#ifdef CPL_IMPL

void line_create(line_t *l, vec2f start, vec2f end, color_t color) {
    l->start = start;
    l->end = end;
    l->color = color;

    float vertices[6] = {
        start.x, start.y, 0.0f, 
        end.x,   end.y,   0.0f, 
    };

    glGenVertexArrays(1, &l->vao);
    glGenBuffers(1, &l->vbo);
    glBindVertexArray(l->vao);
    glBindBuffer(GL_ARRAY_BUFFER, l->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), NULL);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
void line_destroy(line_t *l) {
    if (l->vao != 0 && glIsVertexArray(l->vao)) {
        glDeleteVertexArrays(1, &l->vao);
        l->vao = 0;
    }
    if (l->vbo != 0 && glIsBuffer(l->vbo)) {
        glDeleteBuffers(1, &l->vbo);
        l->vbo = 0;
    }
}
void line_draw_raw(shader_t *s, line_t *l) {
    mat4f transform;
    mat4f_identity(&transform);
    mat4f_translate(&transform, (vec3f){0.0f, 0.0f, 0.0f});

    shader_set_mat4f(s, "transform", &transform);
    shader_set_color(s, "input_color", l->color);
    glBindVertexArray(l->vao);
    glDrawArrays(GL_LINES, 0, 2);
    glBindVertexArray(0);
}

#endif

#pragma endregion

#pragma region Texture & Texture2D

typedef enum { 
    FILTER_LINEAR = 0, 
    FILTER_NEAREST 
} texture_filtering_t;

typedef struct {
    unsigned int id;
    vec2f size;
} texture_t;

typedef struct {
    vec3f rot;
    vec2f pos;
    vec2f size;
    color_t color;
    texture_t *tex;
    unsigned int vbo, vao, ebo;
} texture2D_t;

void texture_load(texture_t *t, char *path, texture_filtering_t filter);
void texture_unload(texture_t *t);
void texture2D_create(texture2D_t *t, vec2f pos, vec2f size, vec3f rot, color_t color, texture_t *tex);
void texture2D_destroy(texture2D_t *t);
void texture2D_draw_raw(shader_t *s, texture2D_t *t, vec2f pivot);

#ifdef CPL_IMPL

void texture_load(texture_t *t, char *path, texture_filtering_t filter) {
    glGenTextures(1, &t->id);
    glBindTexture(GL_TEXTURE_2D, t->id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
    stbi_set_flip_vertically_on_load(1);
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t *data = stbi_load(path, &width, &height, &channels, 0);
    GLenum format = 0;
    if (channels == 1) {
        format = GL_RED;
    } else if (channels == 3) {
        format = GL_RGB;
    } else if (channels == 4) {
        format = GL_RGBA;
    }
    if (data) {
        t->size.x = (float)width;
        t->size.y = (float)height;
        glTexImage2D(GL_TEXTURE_2D, 0, (int)format, (int)t->size.x, (int)t->size.y, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        cpl_log(LOG_ERR, "Failed to load texture");
    }
    stbi_image_free(data);
}
void texture_unload(texture_t *t) {
    if (t->id != 0) {
        glDeleteTextures(1, &t->id);
    }
}
void texture2D_create(texture2D_t *t, vec2f pos, vec2f size, vec3f rot, color_t color, texture_t *tex) {
    t->pos = pos;
    t->size = size;
    t->rot = rot;
    t->color = color;
    t->tex = tex;

    float vertices[22] = {
        size.x, 0.0f,   0.0f, 1.0f, 1.0f, 
        size.x, size.y, 0.0f, 1.0f, 0.0f, 
        0.0f,   size.y, 0.0f, 0.0f, 0.0f, 
        0.0f,   0.0f,   0.0f, 0.0f, 1.0f  
    };
    unsigned int indices[6] = {
        0, 1, 3, 
        1, 2, 3  
    };

    glGenVertexArrays(1, &t->vao);
    glGenBuffers(1, &t->vbo);
    glGenBuffers(1, &t->ebo);
    glBindVertexArray(t->vao);
    glBindBuffer(GL_ARRAY_BUFFER, t->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, t->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), NULL);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
void texture2D_destroy(texture2D_t *t) {
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
    t->tex = NULL;
}
void texture2D_draw_raw(shader_t *s, texture2D_t *t, vec2f pivot) {
    mat4f transform;
    mat4f_identity(&transform);

    mat4f_translate(&transform, (vec3f){t->pos.x, t->pos.y, 0.0f});
    mat4f_translate(&transform, (vec3f){pivot.x, pivot.y, 0.0f});
    if (t->rot.z != 0.0f) {
        mat4f_rotate(&transform, math_rad(t->rot.z), (vec3f){0.0f, 0.0f, 1.0f});
    }
    if (t->rot.y != 0.0f) {
        mat4f_rotate(&transform, math_rad(t->rot.y), (vec3f){0.0f, 1.0f, 0.0f});
    }
    if (t->rot.x != 0.0f) {
        mat4f_rotate(&transform, math_rad(t->rot.x), (vec3f){1.0f, 0.0f, 0.0f});
    }
    mat4f_translate(&transform, (vec3f){-pivot.x, -pivot.y, 0.0f});

    shader_set_int(s, "tex", 0);
    shader_set_mat4f(s, "transform", &transform);
    shader_set_color(s, "input_color", t->color);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t->tex->id);
    glBindVertexArray(t->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#endif

#pragma endregion

#pragma region Text

typedef struct {
    unsigned int id;
    vec2f size;
    vec2f bearing;
    unsigned int advance;
} letter_t;

typedef struct {
    unsigned int vao, vbo;
    char *name;
    letter_t *letters;
} font_t;

void font_load(font_t *f, char *path, char *name, texture_filtering_t filter);
void font_delete(font_t *f);
void text_draw_raw(shader_t *s, font_t *f, char *text, vec2f pos, float scale, color_t color);
vec2f text_get_size(font_t *f, float scale, char *text, ...);

#ifdef CPL_IMPL

void font_load(font_t *f, char *path, char *name, texture_filtering_t filter) {
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

    f->letters = vec_init(f->letters, 128);
    for (uint8_t c = 0; c < 128; c++) {
        if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
            cpl_log(LOG_ERR, "Failed to load Glyph");
            continue;
        }

        unsigned int tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, (GLsizei)face->glyph->bitmap.width, (GLsizei)face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter == FILTER_LINEAR ? GL_LINEAR : GL_NEAREST);

        letter_t character = {
            .id = tex,
            .size = {(float)face->glyph->bitmap.width, (float)face->glyph->bitmap.rows},
            .bearing = {(float)face->glyph->bitmap_left, (float)face->glyph->bitmap_top},
            .advance = face->glyph->advance.x
        };
        vec_push(f->letters, character);
    }
    f->name = name;

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    glGenVertexArrays(1, &f->vao);
    glGenBuffers(1, &f->vbo);
    glBindVertexArray(f->vao);
    glBindBuffer(GL_ARRAY_BUFFER, f->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    FT_Done_Face(face);
    FT_Done_FreeType(ft);
}
void font_delete(font_t *f) {
    if (f->vao != 0 && glIsVertexArray(f->vao)) {
        glDeleteVertexArrays(1, &f->vao);
        f->vao = 0;
    }
    if (f->vbo != 0 && glIsBuffer(f->vbo)) {
        glDeleteBuffers(1, &f->vbo);
        f->vbo = 0;
    }
    for (unsigned int i = 0; i < vec_size(f->letters); i++) {
        glDeleteTextures(1, &f->letters[i].id);
    }
    vec_destroy(f->letters);
}
void text_draw_raw(shader_t *s, font_t *f, char *text, vec2f pos, float scale, color_t color) {
    shader_set_vec3f(s, "text_color", (vec3f){color.r, color.g, color.b});
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(f->vao);

    for (unsigned int i = 0; i < strlen(text); i++) {
        letter_t *l = &f->letters[text[i]];

        float x_pos = pos.x + (l->bearing.x * scale);
        float y_pos = pos.y + ((f->letters['H'].bearing.y - l->bearing.y) * scale);
        float width = l->size.x * scale;
        float height = l->size.y * scale;

        float vertices[6][4] = {
            {x_pos, y_pos + height, 0.0f, 1.0f},
            {x_pos, y_pos, 0.0f, 0.0f},
            {x_pos + width, y_pos, 1.0f, 0.0f},

            {x_pos, y_pos + height, 0.0f, 1.0f},
            {x_pos + width, y_pos, 1.0f, 0.0f},
            {x_pos + width, y_pos + height, 1.0f, 1.0f}
        };

        glBindTexture(GL_TEXTURE_2D, l->id);
        glBindBuffer(GL_ARRAY_BUFFER, f->vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        pos.x += ((float)(l->advance >> 6)) * scale;
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}
vec2f text_get_size(font_t *f, float scale, char *text, ...) {
    float width = 0.0f;
    float height = 0.0f;
    float max_above_base = 0.0f;
    float max_below_base = 0.0f;

    char buffer[KiB(1)];
    if (text) {
        va_list args;
        va_start(args, text);
        vsnprintf(buffer, 1024, text, args);
        va_end(args);
    }

    for (unsigned int i = 0; i < strlen(buffer); i++) {
        letter_t *l = &f->letters[buffer[i]];
        float h = l->size.y * scale;
        max_above_base = math_max(max_above_base, l->bearing.y * scale);
        max_below_base = math_max(max_below_base, (h - (l->bearing.y * scale)));
        width += (float)(l->advance >> 6) * scale;
    }
    height = max_above_base + max_below_base;
    return (vec2f){width, height};
}

#endif

#pragma endregion

#ifdef CPL_INCLUDE_NETWORKING

#pragma region Networking

typedef enum {
    NET_PACKET_RELIABLE = 1,
    NET_PACKET_UNRELIABLE = 8
} net_packet_flag_t;

typedef enum : uint8_t { 
    NET_PACKET_AUDIO_VOICE_MSG = 255 
} net_packet_id_t;

typedef enum : uint8_t {
    NET_CHANNEL_RELIABLE = 0,
    NET_CHANNEL_UNRELIABLE,
    NET_CHANNEL_VOICE_CHAT,
    NET_CHANNELS_SIZE 
} net_channel_t;

#define NET_SEC(s) ((s) * 1000)

#ifdef CPL_IMPL

pthread_mutex_t _net_mutex = PTHREAD_MUTEX_INITIALIZER;

#endif

#pragma region Client

typedef struct {
    ENetHost *client;
    ENetAddress address;
    ENetEvent event;
    ENetPeer *peer;
} client_t;

#ifdef CPL_IMPL

bool _net_worker_running = true;
ENetHost *_host = NULL;
void (*_parse_func)(char *, size_t, net_channel_t, void *) = NULL;
void *_parse_arg = NULL;
pthread_t _net_worker;

#endif

void *_client_net_worker_loop();
void client_create_worker_loop(client_t *client, void (*parse_data)(char *, size_t, net_channel_t, void *), void *parse_data_arg);
void client_destroy_worker_loop();
char *read_ip_from_txt_file(char *path);
bool client_create(client_t *client, char *ip, int port, int wait_ms);
void client_destroy(client_t *client, int wait_ms);

#ifdef CPL_IMPL

void *_client_net_worker_loop() {
    while (_net_worker_running) {
        ENetEvent event;
        pthread_mutex_lock(&_net_mutex);
        if (enet_host_service(_host, &event, 0) > 0) {
            if (event.type == ENET_EVENT_TYPE_RECEIVE) {
                _parse_func((char *)event.packet->data, event.packet->dataLength, event.channelID, _parse_arg);
                enet_packet_destroy(event.packet);
            }
        }

        while (enet_host_check_events(_host, &event) > 0) {
            if (event.type == ENET_EVENT_TYPE_RECEIVE) {
                _parse_func((char *)event.packet->data, event.packet->dataLength, event.channelID, _parse_arg);
                enet_packet_destroy(event.packet);
            }
        }

        pthread_mutex_unlock(&_net_mutex);

        struct timespec ts = {0, 500000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}
void client_create_worker_loop(client_t *client, void (*parse_data)(char *, size_t, net_channel_t, void *), void *parse_data_arg) {
    _host = client->client;
    _parse_func = parse_data;
    _parse_arg = parse_data_arg;
    pthread_create(&_net_worker, NULL, _client_net_worker_loop, NULL);
}
void client_destroy_worker_loop() {
    _net_worker_running = false;
    pthread_join(_net_worker, NULL);
}
char *read_ip_from_txt_file(char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    long file_size = ftell(file);
    if (file_size == -1) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    char *buffer = malloc(file_size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read != file_size) {
        free(buffer);
        fclose(file);
        return NULL;
    }

    buffer[file_size] = '\0';
    buffer[strcspn(buffer, "\r\n")] = '\0';

    fclose(file);
    return buffer;
}
bool client_create(client_t *client, char *ip, int port, int wait_ms) {
    if (enet_initialize()) {
        fprintf(stderr, "Failed to init ENet\n");
        exit(-1);
    }
    atexit(enet_deinitialize);

    client->client = NULL;
    client->client = enet_host_create(NULL, 1, 1, 0, 0);
    if (!client) {
        fprintf(stderr, "Failed to create client host\n");
        exit(-1);
    }

    if (!ip) {
        fprintf(stderr, "IP is invalid\n");
        exit(-1);
    }

    enet_address_set_host(&client->address, ip);
    client->address.port = port;

    client->peer = enet_host_connect(client->client, &client->address, NET_CHANNELS_SIZE, 0);
    if (!client->peer) {
        fprintf(stderr, "No peers available for connection\n");
        exit(-1);
    }

    if (enet_host_service(client->client, &client->event, wait_ms) > 0 && client->event.type == ENET_EVENT_TYPE_CONNECT) {
        return true;
    }
    enet_peer_reset(client->peer);
    return false;
}
void client_destroy(client_t *client, int wait_ms) {
    enet_peer_disconnect(client->peer, 0);
    while (enet_host_service(client->client, &client->event, wait_ms) > 0) {
        if (client->event.type == ENET_EVENT_TYPE_RECEIVE) {
            enet_packet_destroy(client->event.packet);
        } else if (client->event.type == ENET_EVENT_TYPE_DISCONNECT) {
            break;
        }
    }
}

#endif

#pragma endregion

#pragma region Server

typedef struct {
    ENetHost *server;
    ENetAddress address;
} server_t;

void server_init(server_t *server, int port, struct in6_addr host, int max_clients);
void server_destroy(server_t *server);

#ifdef CPL_IMPL

void server_init(server_t *server, int port, struct in6_addr host, int max_clients) {
    if (enet_initialize()) {
        fprintf(stderr, "Failed to init ENet\n");
        exit(-1);
    }
    atexit(enet_deinitialize);

    server->address.host = host;
    server->address.port = port;

    server->server = enet_host_create(&server->address, max_clients, NET_CHANNELS_SIZE, 0, 0);
    if (!server) {
        fprintf(stderr, "Failed to create server host\n");
        exit(-1);
    }
}
void server_destroy(server_t *server) {
    enet_host_destroy(server->server);
}

#endif

#pragma endregion

#pragma region Packets

#pragma region Writer

typedef struct {
    uint8_t data[KiB(5)];
    size_t size;
} packet_writer;

void packet_writer_init(packet_writer *writer, uint8_t packet_id);
void packet_write_char(packet_writer *writer, char val);
void packet_write_int(packet_writer *writer, int val);
void packet_write_uint(packet_writer *writer, unsigned int val);
void packet_write_ulong(packet_writer *writer, unsigned long val);
void packet_write_float(packet_writer *writer, float val);
void packet_write_double(packet_writer *writer, double val);
void packet_write_bool(packet_writer *writer, bool val);
void packet_write_array(packet_writer *writer, void *arr, size_t total_size);
void packet_write_vec2f(packet_writer *writer, vec2f v);

#ifdef CPL_IMPL

void packet_writer_init(packet_writer *writer, uint8_t packet_id) {
    writer->data[0] = packet_id;
    writer->size = 1;
}
void packet_write_char(packet_writer *writer, char val) {
    writer->data[writer->size++] = val;
}
void packet_write_int(packet_writer *writer, int val) {
    memcpy(&writer->data[writer->size], &val, sizeof(int));
    writer->size += sizeof(int);
}
void packet_write_uint(packet_writer *writer, unsigned int val) {
    memcpy(&writer->data[writer->size], &val, sizeof(unsigned int));
    writer->size += sizeof(unsigned int);
}
void packet_write_ulong(packet_writer *writer, unsigned long val) {
    memcpy(&writer->data[writer->size], &val, sizeof(unsigned long));
    writer->size += sizeof(unsigned long);
}
void packet_write_float(packet_writer *writer, float val) {
    memcpy(&writer->data[writer->size], &val, sizeof(float));
    writer->size += sizeof(float);
}
void packet_write_double(packet_writer *writer, double val) {
    memcpy(&writer->data[writer->size], &val, sizeof(double));
    writer->size += sizeof(double);
}
void packet_write_bool(packet_writer *writer, bool val) {
    memcpy(&writer->data[writer->size], &val, sizeof(bool));
    writer->size += sizeof(bool);
}
void packet_write_array(packet_writer *writer, void *arr, size_t total_size) {
    memcpy(&writer->data[writer->size], arr, total_size);
    writer->size += total_size;
}
void packet_write_vec2f(packet_writer *writer, vec2f v) {
    packet_write_float(writer, v.x);
    packet_write_float(writer, v.y);
}

#endif

#pragma endregion

#pragma region Reader

typedef struct {
    uint8_t *data;
    size_t size;
    size_t pos;
} packet_reader;

uint8_t packet_reader_init(packet_reader *reader, uint8_t *data, size_t size);
char packet_read_char(packet_reader *reader);
int packet_read_int(packet_reader *reader);
unsigned int packet_read_uint(packet_reader *reader);
unsigned long packet_read_ulong(packet_reader *reader);
float packet_read_float(packet_reader *reader);
double packet_read_double(packet_reader *reader);
bool packet_read_bool(packet_reader *reader);
void *packet_read_array(packet_reader *reader, size_t total_size);
vec2f packet_read_vec2f(packet_reader *reader);

#ifdef CPL_IMPL

uint8_t packet_reader_init(packet_reader *reader, uint8_t *data, size_t size) {
    reader->data = data;
    reader->size = size;
    reader->pos = 0;
    return reader->data[reader->pos++];
}
char packet_read_char(packet_reader *reader) {
    return (char)reader->data[reader->pos++];
}
int packet_read_int(packet_reader *reader) {
    int val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(int));
    reader->pos += sizeof(int);
    return val;
}
unsigned int packet_read_uint(packet_reader *reader) {
    unsigned int val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(unsigned int));
    reader->pos += sizeof(unsigned int);
    return val;
}
unsigned long packet_read_ulong(packet_reader *reader) {
    unsigned long val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(unsigned long));
    reader->pos += sizeof(unsigned long);
    return val;
}
float packet_read_float(packet_reader *reader) {
    float val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(float));
    reader->pos += sizeof(float);
    return val;
}
double packet_read_double(packet_reader *reader) {
    double val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(double));
    reader->pos += sizeof(double);
    return val;
}
bool packet_read_bool(packet_reader *reader) {
    bool val = 0;
    memcpy(&val, &reader->data[reader->pos], sizeof(bool));
    reader->pos += sizeof(bool);
    return val;
}
void *packet_read_array(packet_reader *reader, size_t total_size) {
    void *arr = malloc(total_size);
    memcpy(arr, &reader->data[reader->pos], total_size);
    reader->pos += total_size;
    return arr;
}
vec2f packet_read_vec2f(packet_reader *reader) {
    return VEC2F(packet_read_float(reader), packet_read_float(reader));
}

#endif

#pragma endregion

void packet_send_to_server(client_t *client, packet_writer *writer, int packet_flag, net_channel_t channel);
void packet_send_to_client(ENetPeer *peer, packet_writer *writer, int packet_flag, net_channel_t channel);
void packet_broadcast(server_t *server, packet_writer *writer, int packet_flag, net_channel_t channel);

#ifdef CPL_IMPL

void packet_send_to_server(client_t *client, packet_writer *writer, int packet_flag, net_channel_t channel) {
    if (client->peer->state != ENET_PEER_STATE_CONNECTED) {
        return;
    }
    pthread_mutex_lock(&_net_mutex);
    ENetPacket *packet = enet_packet_create(writer->data, writer->size, packet_flag);
    if (enet_peer_send(client->peer, channel, packet) < 0) {
        enet_packet_destroy(packet);
    }
    pthread_mutex_unlock(&_net_mutex);
}
void packet_send_to_client(ENetPeer *peer, packet_writer *writer, int packet_flag, net_channel_t channel) {
    ENetPacket *packet = enet_packet_create(writer->data, writer->size, packet_flag);
    enet_peer_send(peer, channel, packet);
}
void packet_broadcast(server_t *server, packet_writer *writer, int packet_flag, net_channel_t channel) {
    ENetPacket *packet = enet_packet_create(writer->data, writer->size, packet_flag);
    enet_host_broadcast(server->server, channel, packet);
}

#endif

#pragma endregion

#pragma endregion

#endif

#pragma region Audio

typedef struct {
    char *path;
    float volume;
    float pitch;
} audio_t;

#ifdef CPL_IMPL

ma_engine _audio_engine;
ma_sound *_music;
ma_sound **_active_sounds;
unsigned int _active_sounds_size;
unsigned int _active_sounds_cap;

bool _muted;
bool _voice_chat_ready;
#ifdef CPL_INCLUDE_NETWORKING
client_t *_client;
#endif
int *_id;
ma_pcm_rb _voice_ring_buffer;
ma_device _playback_device;
ma_device _capture_device;

#endif

void audio_init();
audio_t audio_load(char *path);
void audio_update();
void audio_play_sound(audio_t *a);
void audio_play_music(audio_t *a);
void audio_pause_music();
void audio_resume_music();
void audio_stop_music();
void audio_close();

void _audio_microphone_callback(ma_device *device, void *out, const void *in, unsigned int frame_cnt);
#ifdef CPL_INCLUDE_NETWORKING
void audio_server_broadcast_voice_msg(packet_reader *reader, server_t *server);
void audio_client_handle_voice_msg(packet_reader *reader);
void _audio_playback_callback(ma_device *device, void *out, const void *in, unsigned int frame_cnt);
void audio_muted(bool enable);
void audio_init_voice_chat(client_t *client, int *id);
#endif

#ifdef CPL_IMPL

void audio_init() {
    if (ma_engine_init(NULL, &_audio_engine) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to init audio!");
        exit(-1);
    }
    _active_sounds_cap = 16;
    _active_sounds_size = 0;
    _active_sounds = malloc(_active_sounds_cap * sizeof(ma_sound *));
    _music = NULL;
}
audio_t audio_load(char *path) { return (audio_t){path, 1.0f, 1.0f}; }
void audio_update() {
    unsigned int w = 0;
    for (unsigned int i = 0; i < _active_sounds_size; i++) {
        if (ma_sound_is_playing(_active_sounds[i])) {
            _active_sounds[w++] = _active_sounds[i];
        } else {
            ma_sound_uninit(_active_sounds[i]);
            free(_active_sounds[i]);
        }
    }
    _active_sounds_size = w;
}
void audio_play_sound(audio_t *a) {
    ma_sound *sound = malloc(sizeof(ma_sound));
    if (ma_sound_init_from_file(&_audio_engine, a->path, MA_SOUND_FLAG_DECODE,
                                NULL, NULL, sound) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to init sound!");
        free(sound);
        return;
    }
    ma_sound_set_pitch(sound, a->pitch);
    ma_sound_set_volume(sound, a->volume);
    ma_sound_set_looping(sound, MA_FALSE);
    ma_sound_start(sound);

    if (_active_sounds_size >= _active_sounds_cap) {
        _active_sounds_cap *= 2;
        ma_sound **tmp =
            realloc(_active_sounds, _active_sounds_cap * sizeof(ma_sound *));
        if (!tmp) {
            cpl_log(LOG_ERR, "Realloc sounds failed!");
            return;
        }
        _active_sounds = tmp;
    }
    _active_sounds[_active_sounds_size++] = sound;
}
void audio_play_music(audio_t *a) {
    if (_music) {
        ma_sound_stop(_music);
        ma_sound_uninit(_music);
        free(_music);
        _music = NULL;
    }
    _music = malloc(sizeof(ma_sound));
    if (ma_sound_init_from_file(&_audio_engine, a->path, MA_SOUND_FLAG_DECODE,
                NULL, NULL, _music) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to load music!");
        free(_music);
        _music = NULL;
        return;
    }
    ma_sound_set_pitch(_music, a->pitch);
    ma_sound_set_looping(_music, MA_TRUE);
    ma_sound_start(_music);
}
void audio_pause_music() {
    if (_music) {
        ma_sound_stop(_music);
    }
}
void audio_resume_music() {
    if (_music) {
        ma_sound_start(_music);
    }
}
void audio_stop_music() {
    if (_music) {
        ma_sound_stop(_music);
        ma_sound_seek_to_pcm_frame(_music, 0);
    }
}
void audio_close() {
    audio_update();
    free(_active_sounds);
    if (_music) {
        ma_sound_stop(_music);
        ma_sound_uninit(_music);
        free(_music);
    }
    ma_engine_uninit(&_audio_engine);
}
#ifdef CPL_INCLUDE_NETWORKING
void _audio_microphone_callback(ma_device *device, void *out, const void *in, unsigned int frame_cnt) {
    (void)device;
    (void)out;
    if (_muted || !in) {
        return;
    }

    float *samples = (float *)in;

    packet_writer writer;
    packet_writer_init(&writer, NET_PACKET_AUDIO_VOICE_MSG);
    packet_write_int(&writer, *_id);
    packet_write_uint(&writer, frame_cnt);
    packet_write_array(&writer, samples, frame_cnt * sizeof(float));
    packet_send_to_server(_client, &writer, NET_PACKET_UNRELIABLE, NET_CHANNEL_VOICE_CHAT);
}
void audio_server_broadcast_voice_msg(packet_reader *reader, server_t *server) {
    int id = packet_read_int(reader);
    unsigned int frame_cnt = packet_read_uint(reader);
    float *samples = packet_read_array(reader, frame_cnt * sizeof(float));

    packet_writer writer;
    packet_writer_init(&writer, NET_PACKET_AUDIO_VOICE_MSG);

    packet_write_int(&writer, id);
    packet_write_uint(&writer, frame_cnt);
    packet_write_array(&writer, samples, frame_cnt * sizeof(float));

    packet_broadcast(server, &writer, NET_PACKET_UNRELIABLE, NET_CHANNEL_VOICE_CHAT);

    free(samples);
}
void audio_client_handle_voice_msg(packet_reader *reader) {
    if (!_voice_chat_ready) {
        return;
    }
    int sender_id = packet_read_int(reader);
    if (sender_id == *_id) {
        return;
    }
    unsigned int frame_cnt = packet_read_uint(reader);

    float *samples =
        (float *)packet_read_array(reader, frame_cnt * sizeof(float));
    if (samples == NULL) {
        return;
    }

    unsigned int frames_written = frame_cnt;
    float *rb_buffer = NULL;

    if (ma_pcm_rb_acquire_write(&_voice_ring_buffer, &frames_written,
                                (void **)&rb_buffer) == MA_SUCCESS) {
        if (frames_written > 0 && rb_buffer != NULL) {
            memcpy(rb_buffer, samples, frames_written * sizeof(float));
        }
        ma_pcm_rb_commit_write(&_voice_ring_buffer, frames_written);
    }

    free(samples);
}
void _audio_playback_callback(ma_device *device, void *out, const void *in, unsigned int frame_cnt) {
    (void)in;

    float *out_buffer = (float *)out;
    memset(out_buffer, 0,
           (size_t)frame_cnt * device->playback.channels * sizeof(float));

    unsigned int total_frames_read = 0;

    while (total_frames_read < frame_cnt) {
        unsigned int frames_to_read = frame_cnt - total_frames_read;
        float *rb_buffer = NULL;

        if (ma_pcm_rb_acquire_read(&_voice_ring_buffer, &frames_to_read,
                                   (void **)&rb_buffer) != MA_SUCCESS) {
            break;
        }

        if (frames_to_read == 0 || rb_buffer == NULL) {
            ma_pcm_rb_commit_read(&_voice_ring_buffer, 0);
            break;
        }

        for (unsigned int i = 0; i < frames_to_read; i++) {
            float mono_sample = rb_buffer[i];

            size_t out_idx = (size_t)(total_frames_read + i) * 2;
            out_buffer[out_idx] = mono_sample;
            out_buffer[out_idx + 1] = mono_sample;
        }

        ma_pcm_rb_commit_read(&_voice_ring_buffer, frames_to_read);
        total_frames_read += frames_to_read;
    }
}
void audio_muted(bool enable) { _muted = enable; }
void audio_init_voice_chat(client_t *client, int *id) {
    _client = client;
    _id = id;
    _muted = false;
    ma_pcm_rb_init(ma_format_f32, 1, 48000, NULL, NULL, &_voice_ring_buffer);

    ma_device_config playback_config;

    playback_config = ma_device_config_init(ma_device_type_playback);
    playback_config.playback.format = ma_format_f32;
    playback_config.playback.channels = 2;
    playback_config.sampleRate = 48000;
    playback_config.dataCallback = _audio_playback_callback;
    if (ma_device_init(NULL, &playback_config, &_playback_device) != MA_SUCCESS) {
        cpl_log(LOG_ERR, "Failed to init playback device!");
        return;
    }
    ma_device_start(&_playback_device);

    ma_device_config capture_config;

    capture_config = ma_device_config_init(ma_device_type_capture);
    capture_config.capture.format = ma_format_f32;
    capture_config.capture.channels = 1;
    capture_config.sampleRate = 48000;
    capture_config.dataCallback = _audio_microphone_callback;
    if (ma_device_init(NULL, &capture_config, &_capture_device) != MA_SUCCESS) { 
        cpl_log(LOG_ERR, "Failed to init microphone!");
        return;
    }
    ma_device_start(&_capture_device);

    _voice_chat_ready = true;
}
#endif

#endif

#pragma endregion

#pragma region Screen Quad

typedef struct {
    vec2f size;
    unsigned int vbo, vao, rbo, framebuffer, tex_color_buffer;
} screen_quad_t;

void screen_quad_create(screen_quad_t *q, int width, int height);
void screen_quad_resize(screen_quad_t *q, int width, int height);
void screen_quad_bind(screen_quad_t *q);
void screen_quad_unbind();
void screen_quad_draw(screen_quad_t *q, shader_t *s);

#ifdef CPL_IMPL

void screen_quad_create(screen_quad_t *q, int width, int height) {
    q->size = (vec2f){(float)width, (float)height};

    float vertices[30] = {-1.0f, 1.0f, 0.0f, 0.0f,  1.0f, -1.0f, -1.0f, 0.0f,
                          0.0f,  0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  0.0f,

                          -1.0f, 1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  -1.0f, 0.0f,
                          1.0f,  0.0f, 1.0f, 1.0f,  0.0f, 1.0f,  1.0f};

    glGenVertexArrays(1, &q->vao);
    glGenBuffers(1, &q->vbo);
    glBindVertexArray(q->vao);
    glBindBuffer(GL_ARRAY_BUFFER, q->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), NULL);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));

    glGenFramebuffers(1, &q->framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, q->framebuffer);

    glGenTextures(1, &q->tex_color_buffer);
    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (int)q->size.x, (int)q->size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, q->tex_color_buffer, 0);

    glGenRenderbuffers(1, &q->rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, q->rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, (int)q->size.x, (int)q->size.y);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, q->rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cpl_log(LOG_ERR, "Framebuffer is not complete!");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void screen_quad_resize(screen_quad_t *q, int width, int height) {
    q->size = (vec2f){(float)width, (float)height};

    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, NULL);

    glBindRenderbuffer(GL_RENDERBUFFER, q->rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
void screen_quad_bind(screen_quad_t *q) {
    glBindFramebuffer(GL_FRAMEBUFFER, q->framebuffer);
}
void screen_quad_unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
void screen_quad_draw(screen_quad_t *q, shader_t *s) {
    shader_use(s);
    glBindVertexArray(q->vao);
    glBindTexture(GL_TEXTURE_2D, q->tex_color_buffer);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

#endif

#pragma endregion

#pragma region Collisions

typedef struct {
    vec2f pos;
    vec2f size;
} rect_collider_t;

typedef struct {
    vec2f pos;
    vec2f size;
} triangle_collider_t;

typedef struct {
    vec2f pos;
    float radius;
} circle_collider_t;

bool check_collision_rects(rect_collider_t a, rect_collider_t b);
bool check_collision_circle_rect(circle_collider_t a, rect_collider_t b);
bool check_collision_vec2f_rect(vec2f a, rect_collider_t b);
bool check_collision_circles(circle_collider_t a, circle_collider_t b);
bool check_collision_vec2f_circle(vec2f a, circle_collider_t b);

#ifdef CPL_IMPL

bool check_collision_rects(rect_collider_t a, rect_collider_t b) {
    bool collision_x =
        a.pos.x + a.size.x >= b.pos.x && b.pos.x + b.size.x >= a.pos.x;
    bool collision_y =
        a.pos.y + a.size.y >= b.pos.y && b.pos.y + b.size.y >= a.pos.y;

    return collision_x && collision_y;
}
bool check_collision_circle_rect(circle_collider_t a, rect_collider_t b) {
    vec2f circle_center = a.pos;
    vec2f rect_center =
        vec2f_add(b.pos, VEC2F(b.size.x * 0.5f, b.size.y * 0.5f));
    vec2f half_extents = VEC2F(b.size.x * 0.5f, b.size.y * 0.5f);
    vec2f difference = vec2f_sub(circle_center, rect_center);
    vec2f clamped = vec2f_clamp(difference, VEC2F(-half_extents.x, -half_extents.y), half_extents);
    vec2f closest = vec2f_add(rect_center, clamped);
    vec2f delta = vec2f_sub(closest, circle_center);

    return vec2f_length(delta) <= a.radius;
}
bool check_collision_vec2f_rect(vec2f a, rect_collider_t b) {
    return b.pos.x < a.x && a.x < b.pos.x + b.size.x && b.pos.y < a.y &&
           a.y < b.pos.y + b.size.y;
}
bool check_collision_circles(circle_collider_t a, circle_collider_t b) {
    vec2f dist = vec2f_sub(a.pos, b.pos);
    float distance2 = (dist.x * dist.x) + (dist.y * dist.y);
    float radius_sum = a.radius + b.radius;
    return distance2 <= radius_sum * radius_sum;
}
bool check_collision_vec2f_circle(vec2f a, circle_collider_t b) {
    vec2f dist = vec2f_sub(a, b.pos);
    float distance2 = (dist.x * dist.x) + (dist.y * dist.y);
    return distance2 <= b.radius * b.radius;
}

#endif

#pragma endregion

#pragma region Timing

#ifdef CPL_IMPL

unsigned int _nb_frames = 0;
float _last_time = 0.0f;
float _last_frame = 0.0f;
float _first_frame = 0.0f;
float _dt = 0.0f;
float _time_scale = 1.0f;
float _target_frame_time = 0.0f;
unsigned int _fps = 0;

#endif

float get_time();
void _calc_fps();
unsigned int get_fps();
void _calc_dt();
float get_dt();
float get_time_scale();
void set_time_scale(float scale);
void set_target_fps(unsigned int fps);
void _update_target_fps();

#ifdef CPL_IMPL

float get_time() {
    static struct timespec start_ts;
    static bool initialized = false;
    struct timespec cur_ts;
    if (!initialized) {
        clock_gettime(CLOCK_MONOTONIC, &start_ts);
        initialized = true;
    }
    clock_gettime(CLOCK_MONOTONIC, &cur_ts);
    return (float)((double)(cur_ts.tv_sec - start_ts.tv_sec) + ((double)(cur_ts.tv_nsec - start_ts.tv_nsec) * 1e-9));
}
void _calc_fps() {
    float cur_time = get_time();
    _nb_frames++;
    if (cur_time - _last_time >= 1.0) {
        _fps = _nb_frames;
        _nb_frames = 0;
        _last_time += 1.0f;
    }
}
unsigned int get_fps() { 
    return _fps; 
}
void _calc_dt() {
    float cur_frame = get_time();
    _dt = (cur_frame - _last_frame) * _time_scale;
    _last_frame = cur_frame;
}
float get_dt() { 
    return _dt; 
}
float get_time_scale() { 
    return _time_scale; 
}
void set_time_scale(float scale) { 
    _time_scale = scale; 
}
void set_target_fps(unsigned int fps) {
    _target_frame_time = 1.0f / fps;
}
void _update_target_fps() {
    while (get_time() - _first_frame < _target_frame_time) {}
}

#endif

#pragma endregion

#pragma region Inputs

typedef struct {
    vec2f pos;
    float zoom;
    float rot;
} cam2D_t;

#ifdef CPL_IMPL

GLFWwindow *_window = NULL;
cam2D_t _cam2D;

bool _key_states[KEY_LAST - KEY_SPACE + 1];
bool _prev_key_states[KEY_LAST - KEY_SPACE + 1];
bool _mouse_button_states[MOUSE_BUTTON_LAST + 1];
bool _prev_mouse_button_states[MOUSE_BUTTON_LAST + 1];

#endif

void _input_update();
bool is_key_down(key_button_t key);
bool is_key_up(key_button_t key);
bool is_key_pressed(key_button_t key);
bool is_key_released(key_button_t key);
bool is_mouse_down(mouse_button_t button);
bool is_mouse_pressed(mouse_button_t button);
bool is_mouse_released(mouse_button_t button);
mat4f *cam_2D_get_view_mat(cam2D_t *cam);
cam2D_t *get_cam_2D();
vec2f get_mouse_pos();
vec2f get_screen_to_world_2D(vec2f sp);

#ifdef CPL_IMPL

void _input_update() {
    for (unsigned int i = 0; i < KEY_LAST - KEY_SPACE; i++) {
        _prev_key_states[i] = _key_states[i];
    }
    for (unsigned int key = KEY_SPACE; key <= KEY_LAST; key++) {
        _key_states[key - KEY_SPACE] = glfwGetKey(_window, (int)key) == GLFW_PRESS;
    }

    for (unsigned int i = 0; i < MOUSE_BUTTON_LAST - MOUSE_BUTTON_1; i++) {
        _prev_mouse_button_states[i] = _mouse_button_states[i];
    }
    for (unsigned int button = MOUSE_BUTTON_1; button <= MOUSE_BUTTON_LAST; button++) {
        _mouse_button_states[button - MOUSE_BUTTON_1] = glfwGetMouseButton(_window, (int)button) == GLFW_PRESS;
    }
}
bool is_key_down(key_button_t key) { 
    return _key_states[(int)key - KEY_SPACE]; 
}
bool is_key_up(key_button_t key) { 
    return !_key_states[(int)key - KEY_SPACE]; 
}
bool is_key_pressed(key_button_t key) {
    return _key_states[(int)key - KEY_SPACE] && !_prev_key_states[(int)key - KEY_SPACE];
}
bool is_key_released(key_button_t key) {
    return !_key_states[(int)key - KEY_SPACE] && _prev_key_states[(int)key - KEY_SPACE];
}
bool is_mouse_down(mouse_button_t button) {
    return _mouse_button_states[(int)button - MOUSE_BUTTON_1];
}
bool is_mouse_pressed(mouse_button_t button) {
    return _mouse_button_states[(int)button - MOUSE_BUTTON_1] && !_prev_mouse_button_states[(int)button - MOUSE_BUTTON_1];
}
bool is_mouse_released(mouse_button_t button) {
    return !_mouse_button_states[(int)button - MOUSE_BUTTON_1] && _prev_mouse_button_states[(int)button - MOUSE_BUTTON_1];
}
mat4f *cam_2D_get_view_mat(cam2D_t *cam) {
    mat4f *view = malloc(sizeof(mat4f));
    mat4f_identity(view);

    mat4f_translate(view, (vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    mat4f_translate(view, (vec3f){cam->pos.x, cam->pos.y, 0.0f});
    mat4f_rotate(view, math_rad(cam->rot), (vec3f){0.0f, 0.0f, 1.0f});
    mat4f_translate(view, (vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    mat4f_translate(view, (vec3f){cam->pos.x, cam->pos.y, 0.0f});
    mat4f_scale(view, (vec3f){cam->zoom, cam->zoom, 1.0f});
    mat4f_translate(view, (vec3f){-cam->pos.x, -cam->pos.y, 0.0f});

    return view;
}
cam2D_t *get_cam_2D() { return &_cam2D; }
vec2f get_mouse_pos() {
    double x = 0;
    double y = 0;
    glfwGetCursorPos(_window, &x, &y);
    return VEC2F((float)x, (float)y);
}
vec2f get_screen_to_world_2D(vec2f sp) {
    float x = sp.x;
    float y = sp.y;
    x /= _cam2D.zoom;
    y /= _cam2D.zoom;
    x += _cam2D.pos.x;
    y += _cam2D.pos.y;
    return (vec2f){x, y};
}

#endif

#pragma endregion

#pragma region Window

typedef enum {
    SHAPE_2D_UNLIT = 0,
    SHAPE_2D_LIT,
    TEXT,
    TEXTURE_2D_UNLIT,
    TEXTURE_2D_LIT,
    _DRAW_MODES_COUNT
} draw_mode_t;

#ifndef __EMSCRIPTEN__
#define SHADER_FRAG_PATH(s) "shaders/frag/" s ".frag"
#else
#define SHADER_FRAG_PATH(s) "/shaders/frag/" s "_w.frag"
#endif

#ifndef __EMSCRIPTEN__
#define SHADER_VERT_PATH(s) "shaders/vert/" s ".vert"
#else
#define SHADER_VERT_PATH(s) "/shaders/vert/" s "_w.vert"
#endif

#ifdef CPL_IMPL

GLubyte *_renderer;
GLubyte *_vendor;
GLubyte *_version;
unsigned int _screen_width = 0;
unsigned int _screen_height = 0;
mat4f _projection_2D;
draw_mode_t _cur_draw_mode = SHAPE_2D_UNLIT;
shader_t _shaders[_DRAW_MODES_COUNT];
shader_t _hdr_shader;

#endif

void _framebuffer_size_callback(GLFWwindow *window, int width, int height);
void _web_window_resize();
void _init_shaders();
void window_init(unsigned int width, unsigned int height, char *title, opengl_version_t version);
bool window_should_close();
void window_destroy();
void window_close();
unsigned int get_screen_width();
unsigned int get_screen_height();
vec2f get_screen_size();
void enable_vsync(bool enabled);
void update();
void end_frame();

#ifdef CPL_IMPL

void _framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    (void)window;

    glViewport(0, 0, width, height);
    _screen_width = width;
    _screen_height = height;
    mat4f_ortho(&_projection_2D, 0.0f, (float)_screen_width,
                (float)_screen_height, 0.0f, -1.0f, 1.0f);
}
void _web_window_resize() {
#ifdef __EMSCRIPTEN__
    int w = emscripten_run_script_int("window.innerWidth");
    int h = emscripten_run_script_int("window.innerHeight");
    if ((float)w != (float)_screen_width || (float)h != (float)_screen_height) {
        glfwSetWindowSize(_window, w, h);
    }
#endif
}
void _init_shaders() {
    shader_create(&_shaders[SHAPE_2D_UNLIT], SHADER_VERT_PATH("2D/shape"), SHADER_FRAG_PATH("2D/shape_unlit"));
    shader_create(&_shaders[SHAPE_2D_LIT], SHADER_VERT_PATH("2D/shape"), SHADER_FRAG_PATH("2D/shape_lit"));
    shader_create(&_shaders[TEXT], SHADER_VERT_PATH("2D/text"), SHADER_FRAG_PATH("2D/text"));
    shader_create(&_shaders[TEXTURE_2D_UNLIT], SHADER_VERT_PATH("2D/texture"), SHADER_FRAG_PATH("2D/texture_unlit"));
    shader_create(&_shaders[TEXTURE_2D_LIT], SHADER_VERT_PATH("2D/texture"), SHADER_FRAG_PATH("2D/texture_lit"));
    shader_create(&_hdr_shader, SHADER_VERT_PATH("2D/hdr"), SHADER_FRAG_PATH("2D/hdr"));
}
void window_init(unsigned int width, unsigned int height, char *title, opengl_version_t version) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, (int)version / 10);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, (int)version % 10);
#ifndef __EMSCRIPTEN__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif
#ifdef CPL_INCLUDE_DEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
#else
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 0);
#endif

#ifdef __EMSCRIPTEN__
    double browser_w, browser_h;
    emscripten_get_element_css_size("#canvas", &browser_w, &browser_h);
    width = (int)browser_w;
    height = (int)browser_h;
    if (width <= 0) {
        width = emscripten_run_script_int("window.innerWidth");
    }
    if (height <= 0) {
        height = emscripten_run_script_int("window.innerHeight");
    }
#endif

    _screen_width = width;
    _screen_height = height;

    _window = glfwCreateWindow((int)width, (int)height, title, NULL, NULL);
    if (!_window) {
        cpl_log(LOG_ERR, "Failed to create window");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(_window);
    glfwSetFramebufferSizeCallback(_window, _framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)(glfwGetProcAddress))) {
        cpl_log(LOG_ERR, "Failed to initialize GLAD");
        exit(-1);
    }
#ifdef CPL_INCLUDE_DEBUG
    opengl_debug_enable();
#endif
    _cam2D = (cam2D_t){{0.0f, 0.0f}, 1.0f, 0.0f};
    mat4f_ortho(&_projection_2D, 0.0f, (float)_screen_width, (float)_screen_height, 0.0f, -1.0f, 1.0f);

    _init_shaders();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    _renderer = (GLubyte *)glGetString(GL_RENDERER);
    _vendor = (GLubyte *)glGetString(GL_VENDOR);
    _version = (GLubyte *)glGetString(GL_VERSION);
}
bool window_should_close() { 
    return glfwWindowShouldClose(_window); 
}
void window_destroy() { 
    glfwSetWindowShouldClose(_window, 1); 
}
void window_close() { 
    glfwTerminate(); 
}
unsigned int get_screen_width() { 
    return _screen_width; 
}
unsigned int get_screen_height() { 
    return _screen_height; 
}
vec2f get_screen_size() { 
    return VEC2F(_screen_width, _screen_height); 
}
void enable_vsync(bool enabled) { 
    glfwSwapInterval(enabled); 
}
void update() {
    _first_frame = get_time();
    _calc_fps();
    _calc_dt();
    _input_update();
}
void end_frame() {
    glfwSwapBuffers(_window);
    glfwPollEvents();
    _update_target_fps();
}

#endif

#pragma endregion

#pragma region Drawing

#define NO_ROTATION (vec3f){0.0f, 0.0f, 0.0f}, VEC2F(0.0f, 0.0f)

void clear_background(color_t color);
void begin_draw(draw_mode_t draw_mode, bool mode_2D);
void draw_rect(vec2f pos, vec2f size, color_t color, float rot);
void draw_triangle(vec2f pos, vec2f size, color_t color, float rot);
void draw_circle(vec2f pos, float radius, color_t color);
void draw_line(vec2f start, vec2f end, float thickness, color_t color);
void draw_text(font_t *font, vec2f pos, float scale, color_t color, char *text, ...);
void draw_text_shadow(font_t *font, vec2f pos, float scale, color_t color, vec2f shadow_off, color_t shadow_color, char *text, ...);
void draw_texture2D(texture_t *tex, vec2f pos, vec2f size, color_t color, vec3f rot, vec2f pivot);
void _reset_shader();
void display_details(font_t *font);

#ifdef CPL_IMPL

void clear_background(color_t color) {
    glClearColor(color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}
void begin_draw(draw_mode_t draw_mode, bool mode_2D) {
    _cur_draw_mode = draw_mode;
    shader_use(&_shaders[draw_mode]);

    mat4f view_projection_2D;
    if (mode_2D) {
        mat4f *view = cam_2D_get_view_mat(&_cam2D);
        mat4f_mul(&_projection_2D, view, &view_projection_2D);
        free(view);
    }
    shader_set_mat4f(&_shaders[draw_mode], "projection", mode_2D ? &view_projection_2D : &_projection_2D);
}
void draw_rect(vec2f pos, vec2f size, color_t color, float rot) {
    rect_t r;
    rect_create(&r, pos, size, color, rot);
    rect_draw_raw(&_shaders[_cur_draw_mode], &r);
    rect_destroy(&r);
}
void draw_triangle(vec2f pos, vec2f size, color_t color, float rot) {
    triangle_t t;
    triangle_create(&t, pos, size, color, rot);
    triangle_draw_raw(&_shaders[_cur_draw_mode], &t);
    triangle_destroy(&t);
}
void draw_circle(vec2f pos, float radius, color_t color) {
    circle_t c;
    circle_create(&c, pos, radius, color);
    circle_draw_raw(&_shaders[_cur_draw_mode], &c);
    circle_destroy(&c);
}
void draw_line(vec2f start, vec2f end, float thickness, color_t color) {
    line_t l;
    line_create(&l, start, end, color);
    glLineWidth(thickness);
    line_draw_raw(&_shaders[_cur_draw_mode], &l);
    glLineWidth(1.0f);
    line_destroy(&l);
}
void draw_text(font_t *font, vec2f pos, float scale, color_t color, char *text, ...) {
    char buffer[KiB(1)];
    if (text) {
        va_list args;
        va_start(args, text);
        vsnprintf(buffer, 1024, text, args);
        va_end(args);
    }
    text_draw_raw(&_shaders[_cur_draw_mode], font, buffer, pos, scale, color);
}
void draw_text_shadow(font_t *font, vec2f pos, float scale, color_t color, vec2f shadow_off, color_t shadow_color, char *text, ...) {
    char buffer[KiB(1)];
    if (text) {
        va_list args;
        va_start(args, text);
        vsnprintf(buffer, 1024, text, args);
        va_end(args);
    }
    text_draw_raw(&_shaders[_cur_draw_mode], font, buffer, VEC2F(pos.x + shadow_off.x, pos.y + shadow_off.y), scale, shadow_color);
    text_draw_raw(&_shaders[_cur_draw_mode], font, buffer, pos, scale, color);
}
void draw_texture2D(texture_t *tex, vec2f pos, vec2f size, color_t color, vec3f rot, vec2f pivot) {
    texture2D_t t;
    texture2D_create(&t, pos, size, rot, color, tex);
    texture2D_draw_raw(&_shaders[_cur_draw_mode], &t, pivot);
    texture2D_destroy(&t);
}
void _reset_shader() { 
    shader_use(&_shaders[_cur_draw_mode]); 
}
void display_details(font_t *font) {
    begin_draw(TEXT, false);

    draw_text(font, VEC2F(10.0f, 10.0f), 0.5f, WHITE, "OpenGL version: %s", _version);
    draw_text(font, VEC2F(10.0f, 40.0f), 0.5f, WHITE, "Renderer: %s", _renderer);
    draw_text(font, VEC2F(10.0f, 70.0f), 0.5f, WHITE, "Vendor: %s", _vendor);
    draw_text(font, VEC2F(10.0f, 100.0f), 0.5f, WHITE, "FPS: %d", get_fps());
#ifdef __linux__
    draw_text(font, VEC2F(10.0f, 130.0f), 0.5f, WHITE, "Stack used: %.3f / %.3f MiB (%f%%)",
              (float)profiler_get_stack_used() / 1024.0f / 1024.0f, (float)profiler_get_stack_size() / 1024.0f / 1024.0f,
              (float)profiler_get_stack_used() / (float)profiler_get_stack_size());
    draw_text(font, VEC2F(10.0f, 160.0f), 0.5f, WHITE, "Heap size: %d MiB", profiler_get_heap_size() / 1024 / 1024);
    draw_text(font, VEC2F(10.0f, 190.0f), 0.5f, WHITE, "Heap used: %d MiB", profiler_get_heap_used() / 1024 / 1024);
    draw_text(font, VEC2F(10.0f, 220.0f), 0.5f, WHITE, "Heap free: %d MiB", profiler_get_heap_free() / 1024 / 1024);
#endif
}

#endif

#pragma endregion

#pragma region Lighting

typedef struct {
    vec2f pos;
    float radius;
    float intensity;
    color_t color;
} point_light2D_t;

typedef struct {
    float intensity;
    color_t color;
} global_light2D_t;

void set_ambient_light_2D(float strength);
void set_global_light_2D(global_light2D_t *l);
void add_point_lights_2D(point_light2D_t *ls, unsigned int size);

#ifdef CPL_IMPL

void set_ambient_light_2D(float strength) {
    shader_t *ss = &_shaders[SHAPE_2D_LIT];
    shader_use(ss);
    shader_set_float(ss, "ambient", strength);

    shader_t *ts = &_shaders[TEXTURE_2D_LIT];
    shader_use(ts);
    shader_set_float(ts, "ambient", strength);

    _reset_shader();
}
void set_global_light_2D(global_light2D_t *l) {
    shader_t *ss = &_shaders[SHAPE_2D_LIT];
    shader_use(ss);
    shader_set_float(ss, "g_light.intensity", l->intensity);
    shader_set_color(ss, "g_light.color", l->color);

    shader_t *ts = &_shaders[TEXTURE_2D_LIT];
    shader_use(ts);
    shader_set_float(ts, "g_light.intensity", l->intensity);
    shader_set_color(ts, "g_light.color", l->color);

    _reset_shader();
}
void add_point_lights_2D(point_light2D_t *ls, unsigned int size) {
    shader_t *ss = &_shaders[SHAPE_2D_LIT];
    shader_use(ss);

    shader_set_int(ss, "point_lights_cnt", (int)size);
    for (unsigned int i = 0; i < size; i++) {
        char pos[50];
        snprintf(pos, 50, "point_lights[%d].pos", i);
        char radius[50];
        snprintf(radius, 50, "point_lights[%d].r", i);
        char intensity[50];
        snprintf(intensity, 50, "point_lights[%d].intensity", i);
        char color[50];
        snprintf(color, 50, "point_lights[%d].color", i);

        shader_set_vec2f(ss, pos, ls[i].pos);
        shader_set_float(ss, radius, ls[i].radius);
        shader_set_float(ss, intensity, ls[i].intensity);
        shader_set_color(ss, color, ls[i].color);
    }

    shader_t *ts = &_shaders[TEXTURE_2D_LIT];
    shader_use(ts);

    shader_set_int(ts, "point_lights_cnt", (int)size);
    for (unsigned int i = 0; i < size; i++) {
        char pos[50];
        snprintf(pos, 50, "point_lights[%d].pos", i);
        char radius[50];
        snprintf(radius, 50, "point_lights[%d].r", i);
        char intensity[50];
        snprintf(intensity, 50, "point_lights[%d].intensity", i);
        char color[50];
        snprintf(color, 50, "point_lights[%d].color", i);

        shader_set_vec2f(ts, pos, ls[i].pos);
        shader_set_float(ts, radius, ls[i].radius);
        shader_set_float(ts, intensity, ls[i].intensity);
        shader_set_color(ts, color, ls[i].color);
    }

    _reset_shader();
}

#endif

#pragma endregion

#pragma region Tilemap

#define UV_INSET_TEXEL_SCALE 0.1f

typedef struct {
    float x, y, z;
    float u, v;
} vertex_t;

typedef struct {
    vertex_t *vertices;
    bool *collidable;
    unsigned int count;
    unsigned int capacity;
    unsigned int vbo;
} tilemap_renderer_t;

typedef struct {
    tilemap_renderer_t renderer;
    vec2f size;
    texture_t tex;
    unsigned int vao;
} tilemap_t;

void tilemap_load_texture(tilemap_t *m, char *path, texture_filtering_t filter);
void tilemap_delete_tile(tilemap_t *m, vec2f pos);
bool tilemap_tile_exists(tilemap_t *m, vec2f pos);
void tilemap_check_collidable_tiles(tilemap_t *m, vec2f size);
bool tilemap_tile_collidable(tilemap_t *m, vec2f pos);
vec2f tilemap_get_tile_uv(tilemap_t *m, vec2f pos);
void tilemap_create(tilemap_t *m, vec2f tile_size);
void tilemap_destroy(tilemap_t *m);
void tilemap_begin_editing(tilemap_t *m);
void tilemap_add_tile(tilemap_t *m, vec2f pos, vec2f size, vec2f uv);
void tilemap_draw(tilemap_t *m, color_t color);

#ifdef CPL_IMPL

void tilemap_load_texture(tilemap_t *m, char *path, texture_filtering_t filter) {
    texture_load(&m->tex, path, filter);
}
void tilemap_delete_tile(tilemap_t *m, vec2f pos) {
    for (unsigned int i = 0; i < m->renderer.count; i += 6) {
        if (m->renderer.vertices[i].x == pos.x &&
            m->renderer.vertices[i].y == pos.y) {
            unsigned int right_count = m->renderer.count - (i + 6);
            if (right_count > 0) {
                memmove(&m->renderer.vertices[i], &m->renderer.vertices[i + 6],
                        right_count * sizeof(vertex_t));
                memmove(&m->renderer.collidable[i / 6],
                        &m->renderer.collidable[(i / 6) + 1],
                        (right_count / 6) * sizeof(bool));
            }
            m->renderer.count -= 6;
            break;
        }
    }
}
bool tilemap_tile_exists(tilemap_t *m, vec2f pos) {
    for (unsigned int i = 0; i < m->renderer.count; i += 6) {
        if (m->renderer.vertices[i].x == pos.x &&
            m->renderer.vertices[i].y == pos.y) {
            return true;
        }
    }
    return false;
}
void tilemap_check_collidable_tiles(tilemap_t *m, vec2f size) {
    for (unsigned int i = 0; i < m->renderer.count; i += 6) {
        bool exposed = false;

        vec2f neighbors[4] = {
            {m->renderer.vertices[i].x - size.x, m->renderer.vertices[i].y},
            {m->renderer.vertices[i].x + size.x, m->renderer.vertices[i].y},
            {m->renderer.vertices[i].x, m->renderer.vertices[i].y - size.y},
            {m->renderer.vertices[i].x, m->renderer.vertices[i].y + size.y}};

        for (unsigned int n = 0; n < 4; n++) {
            if (!tilemap_tile_exists(m, neighbors[n])) {
                exposed = true;
                break;
            }
        }
        m->renderer.collidable[i / 6] = exposed;
    }
}
bool tilemap_tile_collidable(tilemap_t *m, vec2f pos) {
    for (unsigned int i = 0; i < m->renderer.count; i += 6) {
        if (m->renderer.vertices[i].x == pos.x &&
            m->renderer.vertices[i].y == pos.y) {
            return m->renderer.collidable[i / 6];
        }
    }
    return false;
}
vec2f tilemap_get_tile_uv(tilemap_t *m, vec2f pos) {
    for (unsigned int i = 0; i < m->renderer.count; i += 6) {
        if (m->renderer.vertices[i].x == pos.x &&
            m->renderer.vertices[i].y == pos.y) {
            float tw = m->size.x / m->tex.size.x;
            float th = m->size.y / m->tex.size.y;
            return VEC2F(m->renderer.vertices[i].u / tw,
                         ((1.0f - m->renderer.vertices[i].v) / th) - 1.0f);
        }
    }
    return VEC2F(-1, -1);
}
void tilemap_create(tilemap_t *m, vec2f tile_size) {
    m->size = tile_size;
    m->renderer.count = 0;
    m->renderer.capacity = 100 * 6;
    m->renderer.vertices = malloc(sizeof(vertex_t) * m->renderer.capacity);
    m->renderer.collidable = malloc(sizeof(bool) * (m->renderer.capacity / 6));

    glGenVertexArrays(1, &m->vao);
    glGenBuffers(1, &m->renderer.vbo);

    glBindVertexArray(m->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m->renderer.vbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t),
                          (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
void tilemap_destroy(tilemap_t *m) {
    free(m->renderer.vertices);
    free(m->renderer.collidable);
    glDeleteBuffers(1, &m->renderer.vbo);
    glDeleteVertexArrays(1, &m->vao);
}
void tilemap_begin_editing(tilemap_t *m) { 
    m->renderer.count = 0; 
}
void tilemap_add_tile(tilemap_t *m, vec2f pos, vec2f size, vec2f uv) {
    if (tilemap_tile_exists(m, pos)) {
        return;
    }
    if (m->renderer.count + 6 > m->renderer.capacity) {
        m->renderer.capacity *= 2;
        vertex_t *tmp_vertices = realloc(m->renderer.vertices,
                                       sizeof(vertex_t) * m->renderer.capacity);
        bool *tmp_collidable = realloc(
            m->renderer.collidable, sizeof(bool) * (m->renderer.capacity / 6));
        if (tmp_vertices && tmp_collidable) {
            m->renderer.vertices = tmp_vertices;
            m->renderer.collidable = tmp_collidable;
        }
    }

    float texel_u = 1.0f / m->tex.size.x;
    float texel_v = 1.0f / m->tex.size.y;

    float inset_u = UV_INSET_TEXEL_SCALE * texel_u;
    float inset_v = UV_INSET_TEXEL_SCALE * texel_v;

    float tw = m->size.x / m->tex.size.x;
    float th = m->size.y / m->tex.size.y;
    float u_start = uv.x * tw;
    float v_start = 1.0f - ((uv.y + 1.0f) * th);

    float u_end = u_start + tw;
    float v_end = v_start + th;

    u_start += inset_u;
    v_start += inset_v;
    u_end   -= inset_u;
    v_end   -= inset_v;

    float x = pos.x;
    float y = pos.y;
    float w = size.x;
    float h = size.y;

    vertex_t quad[6] = {
        {x,     y,     0, u_start, v_start},     
        {x + w, y,     0, u_end, v_start  },   
        {x + w, y + h, 0, u_end, v_end    }, 

        {x,     y,     0, u_start, v_start},     
        {x + w, y + h, 0, u_end, v_end    }, 
        {x,     y + h, 0, u_start, v_end  }    
    };

    memcpy(&m->renderer.vertices[m->renderer.count], quad, sizeof(quad));
    m->renderer.count += 6;
}
void tilemap_draw(tilemap_t *m, color_t color) {
    if (m->renderer.count == 0) {
        return;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m->renderer.vbo);
    glBufferData(GL_ARRAY_BUFFER, (unsigned int)(m->renderer.count * sizeof(vertex_t)), m->renderer.vertices, GL_STREAM_DRAW);

    mat4f transform;
    mat4f_identity(&transform);
    shader_set_mat4f(&_shaders[_cur_draw_mode], "transform", &transform);
    shader_set_color(&_shaders[_cur_draw_mode], "input_color", color);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m->tex.id);

    glBindVertexArray(m->vao);
    glDrawArrays(GL_TRIANGLES, 0, (int)m->renderer.count);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

#endif

#pragma endregion

#pragma region Particle System

#define UNLIMITED_PARTICLES 0

#define PARTICLE(pos, size, dir, color, life_time, rot, tex)                   \
    (particle) { tex, pos, size, dir, color, 0, life_time, rot, true }

typedef struct {
    texture_t *tex;
    vec2f pos;
    vec2f size;
    vec2f dir;
    color_t color;
    float cur_life_time;
    float life_time;
    float rot;
    bool active;
} particle_t;

typedef struct {
    vec2f pos;
    unsigned int max_particles;

    particle_t *particles;
} particle_system_t;

void particle_system_create(particle_system_t *ps, vec2f pos, unsigned int max_particles);
void particle_system_destroy(particle_system_t *ps);
void particle_system_update(particle_system_t *ps);
void particle_system_draw(particle_system_t *ps);
void particle_system_add_particle(particle_system_t *ps, particle_t p);

#ifdef CPL_IMPL

void particle_system_create(particle_system_t *ps, vec2f pos, unsigned int max_particles) {
    ps->pos = pos;
    ps->max_particles = max_particles;
    ps->particles = vec_init(ps->particles, (max_particles >= 10 || max_particles == UNLIMITED_PARTICLES) ? 10 : max_particles);
}
void particle_system_destroy(particle_system_t *ps) {
    vec_destroy(ps->particles);
}
void particle_system_update(particle_system_t *ps) {
    foreach_vec(p, ps->particles) {
        p->cur_life_time += get_dt();
        p->pos =
            vec2f_add(VEC2F(p->dir.x * get_dt(), p->dir.y * get_dt()), p->pos);
        if (p->cur_life_time >= p->life_time) {
            p->active = false;
        }
    }

    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < vec_size(ps->particles); read_idx++) {
        if (ps->particles[read_idx].active) {
            ps->particles[write_idx] = ps->particles[read_idx];
            write_idx++;
        }
    }
    vec_header(ps->particles)->size = write_idx;
}
void particle_system_draw(particle_system_t *ps) {
    foreach_vec(p, ps->particles) {
        draw_texture2D(p->tex, p->pos, p->size, p->color, (vec3f){0.0f, 0.0f, p->rot}, p->pos);
    }
}
void particle_system_add_particle(particle_system_t *ps, particle_t p) {
    if (vec_size(ps->particles) < ps->max_particles || ps->max_particles == 0) {
        vec_push(ps->particles, p);
    }
}

#endif

#pragma endregion

#pragma region HDR

typedef struct {
    unsigned int fbo;
    unsigned int rbo_depth;
    unsigned int color_buffer;
    unsigned int quad_vao;
    unsigned int quad_vbo;
} hdr_t;

#ifdef CPL_IMPL

hdr_t _hdr = (hdr_t){0, 0, 0, 0, 0};

#endif

void hdr_init();
void hdr_quad_resize(hdr_t *h, int width, int height);
void hdr_begin();
void hdr_apply(bool gamma_correct, float exposure);

#ifdef CPL_IMPL

void hdr_init() {
    glGenFramebuffers(1, &_hdr.fbo);
    glGenTextures(1, &_hdr.color_buffer);
    glBindTexture(GL_TEXTURE_2D, _hdr.color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, (int)_screen_width, (int)_screen_height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenRenderbuffers(1, &_hdr.rbo_depth);
    glBindRenderbuffer(GL_RENDERBUFFER, _hdr.rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, (int)_screen_width, (int)_screen_height);
    glBindFramebuffer(GL_FRAMEBUFFER, _hdr.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _hdr.color_buffer, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _hdr.rbo_depth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        cpl_log(LOG_ERR, "Framebuffer is not complete!");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void hdr_quad_resize(hdr_t *h, int width, int height) {
    glBindTexture(GL_TEXTURE_2D, h->color_buffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, NULL);

    glBindRenderbuffer(GL_RENDERBUFFER, h->rbo_depth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}
void hdr_begin() {
    glBindFramebuffer(GL_FRAMEBUFFER, _hdr.fbo);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    hdr_quad_resize(&_hdr, (int)_screen_width, (int)_screen_height);
}
void hdr_apply(bool gamma_correct, float exposure) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shader_use(&_hdr_shader);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _hdr.color_buffer);
    shader_set_bool(&_hdr_shader, "gamma_correct", gamma_correct);
    shader_set_float(&_hdr_shader, "exposure", exposure);
    if (_hdr.quad_vao == 0) {
        float vertices[] = {
            -1.0f, 1.0f,  0.0f, 0.0f, 1.0f, 
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 
            1.0f,  1.0f,  0.0f, 1.0f, 1.0f, 
            1.0f,  -1.0f, 0.0f, 1.0f, 0.0f, 
        };
        glGenVertexArrays(1, &_hdr.quad_vao);
        glGenBuffers(1, &_hdr.quad_vbo);
        glBindVertexArray(_hdr.quad_vao);
        glBindBuffer(GL_ARRAY_BUFFER, _hdr.quad_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), NULL);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    }
    glBindVertexArray(_hdr.quad_vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

#endif

#pragma endregion

#ifdef __cplusplus
}
}
#endif
