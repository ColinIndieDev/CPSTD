#pragma once

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cglm/cglm.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cpstd/cpbase.h"

// {{{ Logging

typedef enum { LOG_INFO, LOG_WARN, LOG_ERR } log_level;

void cpl_log(log_level level, char *message) {
    switch (level) {
    case LOG_INFO:
        printf("[CPL] [INFO]: %s", message);
        break;
    case LOG_WARN:
        printf("[CPL] [WARNING]: %s", message);
        break;
    case LOG_ERR:
        fprintf(stderr, "[CPL] [ERROR]: %s", message);
        break;
    }
}

// }}}

// {{{ Shader

typedef struct {
    u32 id;
} shader;

bool cpl_check_shader_compile_errors(u32 shader, char *type) {
    i32 success = 0;
    char infoLog[1024];

    if (strcmp(type, "PROGRAM") == 0) {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            fprintf(stderr, "[CPL] [ERROR] Program linking error:\n%s\n",
                    infoLog);
            return false;
        }
    } else {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            fprintf(stderr, "[CPL] [ERROR] Shader compilation error: %s\n%s\n",
                    type, infoLog);
            return false;
        }
    }
    return true;
}

char *cpl_read_shader_file(char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    u32 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return NULL;
    }

    u32 read = fread(buffer, 1, size, f);
    fclose(f);

    if (read != size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    return buffer;
}

shader *cpl_create_shader(char *vert_path, char *frag_path) {
    shader *s = malloc(sizeof(shader));

    char *vert_code = cpl_read_shader_file(vert_path);
    char *frag_code = cpl_read_shader_file(frag_path);

    u32 vert = 0;
    u32 frag = 0;
    vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, (const GLchar *const *)&vert_code, NULL);
    glCompileShader(vert);
    cpl_check_shader_compile_errors(vert, "VERTEX");
    frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, (const GLchar *const *)&frag_code, NULL);
    glCompileShader(frag);
    cpl_check_shader_compile_errors(frag, "FRAGMENT");
    s->id = glCreateProgram();
    glAttachShader(s->id, vert);
    glAttachShader(s->id, frag);
    glLinkProgram(s->id);
    cpl_check_shader_compile_errors(s->id, "PROGRAM");

    glDeleteShader(vert);
    glDeleteShader(frag);

    return s;
}

void cpl_use_shader(shader *s) { glUseProgram(s->id); }

void cpl_shader_set_b8(shader *s, char *name, b8 val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_i32(shader *s, char *name, i32 val) {
    glUniform1i(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_f32(shader *s, char *name, f32 val) {
    glUniform1f(glGetUniformLocation(s->id, name), val);
}

void cpl_shader_set_rgba(shader *s, char *name, vec4 *c) {
    glUniform4f(glGetUniformLocation(s->id, name), (*c)[0] / 255.0f,
                (*c)[1] / 255.0f, (*c)[2] / 255.0f, (*c)[3] / 255.0f);
}

void cpl_shader_set_mat4f(shader *s, char *name, mat4 mat) {
    glUniformMatrix4fv(glGetUniformLocation(s->id, name), 1, GL_FALSE,
                       (const GLfloat *)mat);
}

void cpl_shader_set_vec2f(shader *s, char *name, vec2 *v) {
    glUniform2f(glGetUniformLocation(s->id, name), (*v)[0], (*v)[1]);
}

void cpl_shader_set_vec3f(shader *s, char *name, vec3 *v) {
    glUniform3f(glGetUniformLocation(s->id, name), (*v)[0], (*v)[1], (*v)[2]);
}

// }}}

// {{{ Rectangle

typedef struct {
    vec2 pos;
    vec2 size;
    vec4 color;
    f32 rot;

    u32 vbo, vao, ebo;
} rect;

void cpl_create_rect(rect *r, vec2 *pos, vec2 *size, vec4 *color, f32 rot) {
    glm_vec2_copy(*pos, r->pos);
    glm_vec2_copy(*size, r->size);
    glm_vec4_copy(*color, r->color);
    r->rot = rot;

    f32 vertices[12] = {(*size)[0], 0.0f, 0.0f, (*size)[0],
                        (*size)[1], 0.0f, 0.0f, (*size)[1],
                        0.0f,       0.0f, 0.0f, 0.0f};
    u32 indices[6] = {0, 1, 3, 1, 2, 3};

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
                          (void *)NULL);
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
    mat4 transform;
    glm_mat4_identity(transform);

    vec3 pos3 = {r->pos[0], r->pos[1], 0.0f};
    glm_translate(transform, pos3);

    vec3 center3 = {r->size[0] * 0.5f, r->size[1] * 0.5f, 0.0f};
    vec3 neg_center3 = {-r->size[0] * 0.5f, -r->size[1] * 0.5f, 0.0f};
    glm_translate(transform, center3);
    glm_rotate(transform, glm_rad(r->rot), (vec3){0.0f, 0.0f, 1.0f});
    glm_translate(transform, neg_center3);

    cpl_shader_set_mat4f(s, "transform", transform);

    cpl_shader_set_rgba(s, "input_color", &r->color);

    glBindVertexArray(r->vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
}

// }}}

// {{{ Triangle

typedef struct {
    vec2 pos;
    vec2 size;
    vec4 color;
    f32 rot;

    u32 vbo, vao;
} triangle;

void cpl_create_triangle(triangle *t, vec2 *pos, vec2 *size, vec4 *color,
                         f32 rot) {
    glm_vec2_copy(*pos, t->pos);
    glm_vec2_copy(*size, t->size);
    glm_vec4_copy(*color, t->color);
    t->rot = rot;

    f32 vertices[9] = {
        0.0f,       0.0f, 0.0f, (*size)[0], 0.0f, 0.0f, (*size)[0] / 2.0f,
        (*size)[1], 0.0f};

    glGenVertexArrays(1, &t->vao);
    glGenBuffers(1, &t->vbo);
    glBindVertexArray(t->vao);
    glBindBuffer(GL_ARRAY_BUFFER, t->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(f32),
                          (void *)NULL);
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
    mat4 transform;
    glm_mat4_identity(transform);

    vec3 pos3 = {t->pos[0], t->pos[1], 0.0f};
    glm_translate(transform, pos3);

    vec3 center3 = {t->size[0] * 0.5f, t->size[1] * 0.5f, 0.0f};
    vec3 neg_center3 = {-t->size[0] * 0.5f, -t->size[1] * 0.5f, 0.0f};
    glm_translate(transform, center3);
    glm_rotate(transform, glm_rad(t->rot), (vec3){0.0f, 0.0f, 1.0f});
    glm_translate(transform, neg_center3);

    cpl_shader_set_mat4f(s, "transform", transform);

    cpl_shader_set_rgba(s, "input_color", &t->color);

    glBindVertexArray(t->vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

// }}}

// {{{ General

f32 cpl_screen_width = 0.0f;
f32 cpl_screen_height = 0.0f;

GLFWwindow *cpl_window = NULL;

mat4 cpl_projection_2D;

typedef enum { CPL_SHAPE_2D_UNLIT, CPL_DRAW_MODES_COUNT } cpl_draw_mode;

cpl_draw_mode cpl_cur_draw_mode;

shader *cpl_shaders = NULL;

i32 cpl_nb_frames = 0;
f32 cpl_last_time = 0.0f;
f32 cpl_last_frame = 0.0f;
f32 cpl_dt = 0.0f;
f32 cpl_time_scale = 1.0f;
i32 cpl_fps = 0;

typedef struct {
    vec2 pos;
    f32 zoom;
    f32 rot;
} cam_2D;

cam_2D cpl_cam_2D;

mat4 *cpl_cam_2D_get_view_mat(cam_2D *cam) {
    mat4 *view = malloc(sizeof(mat4));
    glm_mat4_identity(*view);

    glm_translate(*view, (vec3){-cam->pos[0], -cam->pos[1], 0.0f});

    glm_translate(*view, (vec3){cam->pos[0], cam->pos[1], 0.0f});
    glm_rotate(*view, glm_rad(cam->rot), (vec3){0.0f, 0.0f, 1.0f});
    glm_translate(*view, (vec3){-cam->pos[0], -cam->pos[1], 0.0f});

    glm_translate(*view, (vec3){cam->pos[0], cam->pos[1], 0.0f});
    glm_scale(*view, (vec3){cam->zoom, cam->zoom, 1.0f});
    glm_translate(*view, (vec3){-cam->pos[0], -cam->pos[1], 0.0f});

    return view;
}

// {{{ Window

void cpl_framebuffer_size_callback(GLFWwindow *window, i32 width, i32 height) {
    glViewport(0, 0, width, height);
    cpl_screen_width = (f32)width;
    cpl_screen_height = (f32)height;
    glm_ortho(0.0f, cpl_screen_width, cpl_screen_height, 0.0f, -1.0f, 1.0f,
              cpl_projection_2D);
}

shader *cpl_init_shaders() {
    shader *shaders = malloc(sizeof(shader) * CPL_DRAW_MODES_COUNT);

    shaders[0] = *cpl_create_shader("shaders/vert/2D/shape_unlit.vert",
                                    "shaders/frag/2D/shape_unlit.frag");

    return shaders;
}
void cpl_init_window(i32 width, i32 height, char *title) {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 0);

    cpl_screen_width = (f32)width;
    cpl_screen_height = (f32)height;

    cpl_window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (cpl_window == NULL) {
        fprintf(stderr, "[CPL] [ERROR] Failed to create window\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(cpl_window);
    glfwSetFramebufferSizeCallback(cpl_window, cpl_framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "[CPL] [ERROR] Failed to initialize GLAD");
        exit(-1);
    }

    cpl_cam_2D = (cam_2D){{0.0f, 0.0f}, 1.0f, 0.0f};
    glm_ortho(0.0f, cpl_screen_width, cpl_screen_height, 0.0f, -1.0f, 1.0f,
              cpl_projection_2D);

    cpl_shaders = cpl_init_shaders();
}

b8 cpl_window_should_close() { return glfwWindowShouldClose(cpl_window); }

void cpl_destroy_window() { glfwSetWindowShouldClose(cpl_window, 1); }

void cpl_close_window() { glfwTerminate(); }

// }}}

// {{{ Drawing

void cpl_clear_background(f32 r, f32 g, f32 b, f32 a) {
    glClearColor(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void cpl_begin_draw(cpl_draw_mode draw_mode, b8 mode_2D) {
    cpl_cur_draw_mode = draw_mode;
    cpl_use_shader(&cpl_shaders[draw_mode]);
    if (mode_2D) {

    } else {
        mat4 view_projection_2D;
        if (mode_2D) {
            glm_mat4_mul(cpl_projection_2D,
                         *cpl_cam_2D_get_view_mat(&cpl_cam_2D),
                         view_projection_2D);
        }
        cpl_shader_set_mat4f(&cpl_shaders[draw_mode], "projection",
                             mode_2D ? view_projection_2D : cpl_projection_2D);
    }
}

void cpl_draw_rect(vec2 *pos, vec2 *size, vec4 *color, f32 rot) {
    rect r = {**pos, **size, **color, rot, 0, 0, 0};
    cpl_create_rect(&r, pos, size, color, rot);
    cpl_draw_rect_raw(&cpl_shaders[cpl_cur_draw_mode], &r);
    cpl_destroy_rect(&r);
}

void cpl_draw_triangle(vec2 *pos, vec2 *size, vec4 *color, f32 rot) {
    triangle t = {**pos, **size, **color, rot, 0, 0, 0};
    cpl_create_triangle(&t, pos, size, color, rot);
    cpl_draw_triangle_raw(&cpl_shaders[cpl_cur_draw_mode], &t);
    cpl_destroy_triangle(&t);
}

// }}}

void cpl_calc_fps() {
    f32 cur_time = (f32)glfwGetTime();
    cpl_nb_frames++;
    if (cur_time - cpl_last_time >= 1.0) {
        cpl_fps = cpl_nb_frames;
        cpl_nb_frames = 0;
        cpl_last_time += 1.0f;
    }
}
i32 cpl_get_fps() { return cpl_fps; }

void cpl_calc_dt() {
    f32 cur_frame = (f32)glfwGetTime();
    cpl_dt = (cur_frame - cpl_last_frame) * cpl_time_scale;
    cpl_last_frame = cur_frame;
}

f32 cpl_get_dt() { return cpl_dt; }
f32 cpl_get_time() { return (f32)glfwGetTime(); }
f32 cpl_get_time_scale() { return cpl_time_scale; }
void cpl_set_time_scale(f32 scale) { cpl_time_scale = scale; }
void cpl_enable_vsync(b8 enabled) { glfwSwapInterval(enabled); }

void cpl_update() {
    cpl_calc_fps();
    cpl_calc_dt();
}

void cpl_end_frame() {
    glfwSwapBuffers(cpl_window);
    glfwPollEvents();
}

// }}}
