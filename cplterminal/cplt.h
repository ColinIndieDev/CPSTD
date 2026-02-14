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

typedef struct {
    u8 r, g, b;
} rgb;

struct termios orig_termios;

void cplt_disable_raw_mode() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

void cplt_activate_raw_mode() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(cplt_disable_raw_mode);

    struct termios raw = orig_termios;

    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    flags = fcntl(STDOUT_FILENO, F_GETFL, 0);
    fcntl(STDOUT_FILENO, F_SETFL, flags & ~O_NONBLOCK);
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

#define CPLT_MAX_KEYS 256
#define CPLT_KEY_TIMEOUT 0.15f

bool keyDown[CPLT_MAX_KEYS] = {false};
bool keyPressed[CPLT_MAX_KEYS] = {false};
bool keyReleased[CPLT_MAX_KEYS] = {false};
float keyTimers[CPLT_MAX_KEYS] = {0};

void UpdateInput() {
    float now = cplt_get_time();

    memset(keyPressed, false, sizeof(keyPressed));
    memset(keyReleased, false, sizeof(keyReleased));

    char c;
    while (read(STDIN_FILENO, &c, 1) == 1) {
        u8 uc = (u8)c;

        if (!keyDown[uc]) {
            keyPressed[uc] = true;
        }
        keyDown[uc] = true;
        keyTimers[uc] = now;
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

int cplt_rgb_to_int(int r, int g, int b) { return (r << 16) | (g << 8) | b; }

void cplt_render() {
    printf("\x1b[H");

    char line[4096];
    int lastColor = -1;

    for (size_t y = 0; y < height; y++) {
        int pos = 0;

        pos += snprintf(line + pos, sizeof(line) - pos, "%s", bgColor);

        for (size_t x = 0; x < width; x++) {
            size_t idx = (y * width) + x;
            int packedColor = colorBuffer[idx];

            if (packedColor != lastColor) {
                int r = (packedColor >> 16) & 0xFF;
                int g = (packedColor >> 8) & 0xFF;
                int b = packedColor & 0xFF;

                pos += snprintf(line + pos, sizeof(line) - pos,
                                "\x1b[38;2;%d;%d;%dm", r, g, b);
                lastColor = packedColor;
            }

            char c = screenBuffer[idx];
            if (c < 32 || c == 127) {
                c = ' ';
            }
            line[pos++] = c;
        }

        line[pos++] = '\n';
        write(STDOUT_FILENO, line, pos);
    }

    fflush(stdout);
}

void cplt_clear(char c, int r, int g, int b) {
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            screenBuffer[(y * width) + x] = c;
            colorBuffer[(y * width) + x] = cplt_rgb_to_int(r, g, b);
        }
    }
}

void cplt_clear_bg(int r, int g, int b) {
    r = r < 0 ? 0 : r % 256;
    g = g < 0 ? 0 : g % 256;
    b = b < 0 ? 0 : b % 256;
    snprintf(bgColor, sizeof(bgColor), "\x1b[48;2;%d;%d;%dm", r, g, b);
}

void cplt_draw_pixel(int x, int y, const char *c, int r, int g, int b) {
    if (x < width && x >= 0 && y < height && y >= 0) {
        screenBuffer[(y * width) + x] = c[0];
        colorBuffer[(y * width) + x] = cplt_rgb_to_int(r, g, b);
    }
}

void cplt_draw_text(int x, int y, char *text, int r, int g, int b) {
    int ex = (int)strlen(text) + x;
    if (y >= height || y < 0 || x >= height || ex < 0) {
        return;
    }
    for (size_t ix = x; ix < ex; ix++) {
        cplt_draw_pixel((int)ix, y, &text[ix - x], r, g, b);
    }
}

void cplt_draw_rect(int x, int y, int w, int h, char *c, int r, int g, int b) {
    for (int iy = y; iy < h + y; iy++) {
        if (iy >= height || iy < 0) {
            continue;
        }
        for (int ix = x; ix < w + x; ix++) {
            cplt_draw_pixel(ix, iy, c, r, g, b);
        }
    }
}

void cplt_draw_circle(int cx, int cy, int r, char *c, int red, int g, int b) {
    int x = 0;
    int y = r;
    int d = 1 - r;

    while (x <= y) {
        for (int i = cx - x; i <= cx + x; i++) {
            cplt_draw_pixel(i, cy + y, c, red, g, b);
            cplt_draw_pixel(i, cy - y, c, red, g, b);
        }
        for (int i = cx - y; i <= cx + y; i++) {
            cplt_draw_pixel(i, cy + x, c, red, g, b);
            cplt_draw_pixel(i, cy - x, c, red, g, b);
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

void cplt_draw_circle_out(int cx, int cy, int r, char *c, int red, int g,
                          int b) {
    int x = 0;
    int y = r;
    int d = 1 - r;

    while (x <= y) {
        cplt_draw_pixel(cx + x, cy + y, c, red, g, b);
        cplt_draw_pixel(cx + y, cy + x, c, red, g, b);
        cplt_draw_pixel(cx - x, cy + y, c, red, g, b);
        cplt_draw_pixel(cx - y, cy + x, c, red, g, b);
        cplt_draw_pixel(cx + x, cy - y, c, red, g, b);
        cplt_draw_pixel(cx + y, cy - x, c, red, g, b);
        cplt_draw_pixel(cx - x, cy - y, c, red, g, b);
        cplt_draw_pixel(cx - y, cy - x, c, red, g, b);
        x++;
        if (d < 0) {
            d += (2 * x) + 1;
        } else {
            y--;
            d += (2 * (x - y)) + 1;
        }
    }
}
