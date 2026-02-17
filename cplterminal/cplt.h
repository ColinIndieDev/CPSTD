#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include "../cpstd/cpbase.h"

int width = 0;
int height = 0;
char *screenBuffer;
int *colorBuffer;
char bgColor[22] = "\x1b[48;2;0;0;0m";

bool running = true;

int nbFrames = 0;
int fps = 0;
float lastFrame = 0.0f;
float lastFPS = 0.0f;
float timeScale = 1.0f;
float dt = 0.0f;

#define CPLT_DEF_COLOR "\x1b[0m"
#define WHITE (rgb){255, 255, 255}
#define BLACK (rgb){0, 0, 0}
#define RED (rgb){255, 0, 0}
#define GREEN (rgb){0, 255, 0}
#define BLUE (rgb){0, 0, 255}
#define PINK (rgb){255, 0, 255}

typedef struct {
    u8 r, g, b;
} rgb;

typedef struct {
    f32 x, y, w, h;
} rect;

struct termios orig_termios;

void cplt_disable_raw_mode() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

void cplt_activate_raw_mode() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(cplt_disable_raw_mode);

    struct termios raw = orig_termios;

    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void cplt_hide_cursor(bool hide) {
    if (hide) {
        write(STDOUT_FILENO, "\x1b[?25l", 6);
    } else {
        write(STDOUT_FILENO, "\x1b[?25h", 6);
    }
}

void cplt_init(int w, int h) {
    width = w;
    height = h;
    screenBuffer = malloc((size_t)width * height);
    colorBuffer = malloc((size_t)width * height * sizeof(int));
    memset(screenBuffer, ' ', (size_t)width * height);
    memset(colorBuffer, 0, (size_t)width * height * sizeof(int));

    printf("\x1b[2J");
    cplt_hide_cursor(true);
    cplt_activate_raw_mode();
}

float cplt_get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (float)ts.tv_sec + ((float)ts.tv_nsec / 1e9f);
}

float cplt_get_dt() { return dt; }

void cplt_calc_fps() {
    float curTime = cplt_get_time();
    nbFrames++;
    if (curTime - lastFPS >= 1.0) {
        fps = nbFrames;
        nbFrames = 0;
        lastFPS = curTime;
    }
}

void cplt_calc_dt() {
    float curFrame = cplt_get_time();
    dt = (curFrame - lastFrame) * timeScale;
    lastFrame = curFrame;
}

bool cplt_check_collision_rects(rect *a, rect *b) {
    return (a->x + a->w >= b->x && b->x + b->w >= a->x) &&
           (a->y + a->h >= b->y && b->y + b->h >= a->y);
}

// {{{ Inputs

#define CPLT_MAX_KEYS 256
#define CPLT_KEY_TIMEOUT 0.6f

bool keyDown[CPLT_MAX_KEYS] = {false};
bool keyPressed[CPLT_MAX_KEYS] = {false};
bool keyReleased[CPLT_MAX_KEYS] = {false};
float keyTimers[CPLT_MAX_KEYS] = {0};

void cplt_update_input() {
    float now = cplt_get_time();

    memset(keyPressed, false, sizeof(keyPressed));
    memset(keyReleased, false, sizeof(keyReleased));

    char buf[64];
    int n;
    while ((n = (i32)read(STDIN_FILENO, buf, sizeof(buf))) > 0) {
        for (int i = 0; i < n; i++) {
            u8 uc = (u8)buf[i];
            if (!keyDown[uc]) {
                keyPressed[uc] = true;
            }
            keyDown[uc] = true;
            keyTimers[uc] = now;
        }
    }

    for (int i = 0; i < CPLT_MAX_KEYS; i++) {
        if (keyDown[i] && (now - keyTimers[i] > CPLT_KEY_TIMEOUT)) {
            keyDown[i] = false;
            keyReleased[i] = true;
        }
    }
}

bool cplt_is_key_down(char key) { return keyDown[(u8)key]; }

bool cplt_is_key_pressed(char key) { return keyPressed[(u8)key]; }

bool cplt_is_key_released(char key) { return keyReleased[(u8)key]; }

// }}}

int cplt_rgb_to_int(rgb color) {
    return (color.r << 16) | (color.g << 8) | color.b;
}

// {{{ Drawing

void cplt_render() {
    int lineBufSize = (width * 22) + 64;
    char *line = malloc((size_t)lineBufSize);
    if (!line) {
        return;
    }

    int totalSize = ((lineBufSize + 8) * height) + 64;
    char *out = malloc((size_t)totalSize);
    if (!out) {
        free(line);
        return;
    }
    int outPos = 0;

    outPos += snprintf(out + outPos, (size_t)(totalSize - outPos), "\x1b[H");

    for (int y = 0; y < height; y++) {
        int pos = 0;

        pos += snprintf(line + pos, (size_t)(lineBufSize - pos), "%s", bgColor);
        int lastColor = -1;

        for (int x = 0; x < width; x++) {
            int idx = (y * width) + x;
            int packedColor = colorBuffer[idx];

            if (packedColor != lastColor) {
                int r = (packedColor >> 16) & 0xFF;
                int g = (packedColor >> 8) & 0xFF;
                int b = packedColor & 0xFF;
                pos += snprintf(line + pos, (size_t)(lineBufSize - pos),
                                "\x1b[38;2;%d;%d;%dm", r, g, b);
                lastColor = packedColor;
            }

            char c = screenBuffer[idx];
            if (c < 32 || c == 127) {
                c = ' ';
            }
            line[pos++] = c;
        }

        pos +=
            snprintf(line + pos, (size_t)(lineBufSize - pos), CPLT_DEF_COLOR);

        if (y < height - 1) {
            line[pos++] = '\r';
            line[pos++] = '\n';
        }

        memcpy(out + outPos, line, (size_t)pos);
        outPos += pos;
    }

    write(STDOUT_FILENO, out, (size_t)outPos);
    free(line);
    free(out);
}

void cplt_clear(char c, rgb color) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            screenBuffer[(y * width) + x] = c;
            colorBuffer[(y * width) + x] = cplt_rgb_to_int(color);
        }
    }
}

void cplt_clear_bg(rgb color) {
    color.r = color.r < 0 ? 0 : color.r % 256;
    color.g = color.g < 0 ? 0 : color.g % 256;
    color.b = color.b < 0 ? 0 : color.b % 256;
    snprintf(bgColor, sizeof(bgColor), "\x1b[48;2;%d;%d;%dm", color.r, color.g,
             color.b);
}

void cplt_draw_pixel(int x, int y, const char *c, rgb color) {
    if (x < width && x >= 0 && y < height && y >= 0) {
        screenBuffer[(y * width) + x] = c[0];
        colorBuffer[(y * width) + x] = cplt_rgb_to_int(color);
    }
}

void cplt_draw_text(int x, int y, char *text, rgb color) {
    if (y < 0 || y >= height) {
        return;
    }
    int len = (int)strlen(text);
    if (x >= width || x + len <= 0) {
        return;
    }

    int start = 0;
    if (x < 0) {
        start = -x;
        x = 0;
    }
    int end = len;
    if (x + (end - start) > width) {
        end = start + (width - x);
    }

    for (int i = start; i < end; i++) {
        cplt_draw_pixel(x + (i - start), y, &text[i], color);
    }
}

void cplt_draw_rect(int x, int y, int w, int h, char *c, rgb color) {
    for (int iy = y; iy < h + y; iy++) {
        if (iy >= height || iy < 0) {
            continue;
        }
        for (int ix = x; ix < w + x; ix++) {
            cplt_draw_pixel(ix, iy, c, color);
        }
    }
}

void cplt_draw_circle(int cx, int cy, int r, char *c, rgb color) {
    int x = 0;
    int y = r;
    int d = 1 - r;

    while (x <= y) {
        for (int i = cx - x; i <= cx + x; i++) {
            cplt_draw_pixel(i, cy + y, c, color);
            cplt_draw_pixel(i, cy - y, c, color);
        }
        for (int i = cx - y; i <= cx + y; i++) {
            cplt_draw_pixel(i, cy + x, c, color);
            cplt_draw_pixel(i, cy - x, c, color);
        }
        x++;
        if (d < 0) {
            d += (2 * x) + 1;
        } else {
            y--;
            d += (2 * (x - y)) + 1;
        }
    }
}

void cplt_draw_circle_out(int cx, int cy, int r, char *c, rgb color) {
    int x = 0;
    int y = r;
    int d = 1 - r;

    while (x <= y) {
        cplt_draw_pixel(cx + x, cy + y, c, color);
        cplt_draw_pixel(cx + y, cy + x, c, color);
        cplt_draw_pixel(cx - x, cy + y, c, color);
        cplt_draw_pixel(cx - y, cy + x, c, color);
        cplt_draw_pixel(cx + x, cy - y, c, color);
        cplt_draw_pixel(cx + y, cy - x, c, color);
        cplt_draw_pixel(cx - x, cy - y, c, color);
        cplt_draw_pixel(cx - y, cy - x, c, color);
        x++;
        if (d < 0) {
            d += (2 * x) + 1;
        } else {
            y--;
            d += (2 * (x - y)) + 1;
        }
    }
}

// }}}
