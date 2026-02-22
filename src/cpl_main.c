#include "../cplibrary/cpl.h"

int main() {
    cpl_init_window(800, 600, "Hello CPL");
    cpl_enable_vsync(false);

    while (!cpl_window_should_close()) {
        cpl_update();

        cpl_clear_background(0.0f, 0.0f, 0.0f, 255.0f);

        cpl_begin_draw(CPL_SHAPE_2D_UNLIT, false);

        cpl_draw_rect(&(vec2){0.0f, 0.0f}, &(vec2){100.0f, 100.0f},
                      &(vec4){255.0f, 0.0f, 0.0f, 255.0f}, 0.0f);

        cpl_draw_triangle(&(vec2){100.0f, 100.0f}, &(vec2){100.0f, 100.0f},
                      &(vec4){255.0f, 255.0f, 0.0f, 255.0f}, 90.0f);

        cpl_end_frame();
    }
    cpl_close_window();
}
