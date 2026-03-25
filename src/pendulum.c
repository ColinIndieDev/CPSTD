#define CPL_IMPLEMENTATION
#include "../cplibrary/cpl.h"

#define dt 0.01f
#define D 3.45f
#define m 1.0f
#define g 9.81f
#define k 8.9e-4f /* 1.5e-2 */

VEC_DEF(vec2f, trail_pos)

int main(void) {
    cpl_init_window(800, 800, "Pendulum Simulation");
    cpl_enable_vsync(false);

    f32 x = -0.2f;
    f32 origin = (m * g) / D;
    f32 F = (m * g) + (D * x);
    f32 a = F / m;
    f32 v = 0.0f;
    f32 F_r = k * (v * CPM_ABS(v));

    f32 scale = 100.0f; // m -> cm

    trail_pos trail;
    trail_pos_reserve(&trail, 100);

    cpl_set_time_scale(1.0f);

    while (!cpl_window_should_close()) {
        cpl_update();

        f32 speed = 100.0f;

        if (cpl_is_key_down(CPL_KEY_W)) {
            cpl_cam_2D.pos.y -= speed * cpl_get_dt();
        }
        if (cpl_is_key_down(CPL_KEY_S)) {
            cpl_cam_2D.pos.y += speed * cpl_get_dt();
        }
        if (cpl_is_key_down(CPL_KEY_A)) {
            cpl_cam_2D.pos.x -= speed * cpl_get_dt();
        }
        if (cpl_is_key_down(CPL_KEY_D)) {
            cpl_cam_2D.pos.x += speed * cpl_get_dt();
        }

        if (cpl_is_key_down(CPL_KEY_H)) {
            cpl_cam_2D.zoom *= 1.1f;
        }
        if (cpl_is_key_down(CPL_KEY_N)) {
            cpl_cam_2D.zoom /= 1.1f;
        }

        F_r = k * (v * CPM_ABS(v));
        F = (m * g) - (D * x) - F_r;
        a = F / m;
        v += (a * cpl_get_dt());
        x += (v * cpl_get_dt());

        vec2f ball_pos = {.x = cpl_get_screen_width() / 2.0f,
                          .y = (x * scale) + (cpl_get_screen_height() / 2.0f)};

        f32 anchor_y = (cpl_get_screen_height() / 2.0f) - (origin * scale);

        trail_pos_push_back(&trail, ball_pos);

        cpl_clear_background(&BLACK);

        cpl_begin_draw(CPL_SHAPE_2D_UNLIT, true);

        cpl_draw_line(&(vec2f){ball_pos.x, anchor_y},
                      &ball_pos, 0.5f, &WHITE);

        cpl_draw_circle(&ball_pos, 0.02f * scale, &RED);

        FOREACH_VEC(vec2f, trail_pos, t, &trail) {
            t->x -= 25.0f * cpl_get_dt();
            cpl_draw_circle(t, 1.0f, &(vec4f){255.0f, 255.0f, 0.0f, 50.0f});
        }

        cpl_end_frame();
    }
    cpl_close_window();
}
