#define CPL_IMPL
#include <cpl/cpl.h>

#include <cpstd/rand.h>

vec2f pos = VEC2F(0, 0);

int main(void) {
    window_init(800, 800, "Test", OPENGL_VER_3_3);
    enable_vsync(false);

    pcg_rand_seed();

    while (!window_should_close()) {
        update();
        float speed = 100.0f;
        if (is_key_down(KEY_LETTER_A)) {
            pos.x -= speed * get_dt();
        }
        if (is_key_down(KEY_LETTER_D)) {
            pos.x += speed * get_dt();
        }
        if (is_key_down(KEY_LETTER_W)) {
            pos.y -= speed * get_dt();
        }
        if (is_key_down(KEY_LETTER_S)) {
            pos.y += speed * get_dt();
        }

        clear_background(BLUE);

        begin_draw(SHAPE_2D_UNLIT, false);

        draw_rect(pos, VEC2F(250.0f, 250.0f), PURPLE, 0);

        for (int i = 0; i < 10000; i++) {
            draw_rect(VEC2F(pcg_randf_range(0.0f, (float)get_screen_width()), pcg_randf_range(0.0f, (float)get_screen_height())), VEC2F(10.0f, 10.0f), RED, 0);
        }

        end_frame();

        printf("%d\n", get_fps());
    }
    window_close();
}
