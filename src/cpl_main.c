#include "../cplibrary/cpl.h"

int main() {
    cpl_init_window(800, 600, "Hello CPL");
    cpl_enable_vsync(false);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    while (!cpl_window_should_close()) {
        cpl_update();

        cpl_clear_background(&BLACK);

        cpl_begin_draw(CPL_SHAPE_2D_UNLIT, false);

        cpl_draw_rect(&(vec2f){0.0f, 0.0f}, &(vec2f){100.0f, 100.0f}, &RED, 0.0f);

        cpl_draw_triangle(&(vec2f){100.0f, 100.0f}, &(vec2f){100.0f, 100.0f},
                          &GREEN, cpl_get_time() * 360);

        cpl_draw_line(&(vec2f){0.0f, 0.0f}, &(vec2f){300.0f, 300.0f}, 2.0f,
                      &BLUE);

        cpl_draw_circle(&(vec2f){300.0f, 300.0f}, 100.0f, &WHITE);

        cpl_end_frame();
    }
    cpl_close_window();
}
