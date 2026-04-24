#define CPL_IMPLEMENTATION
#include "../cplibrary/cpl.h"

#include "../cpstd/cprng.h"

void handle_movement() {
    f32 speed = 100.0f;

    if (cpl_is_key_down(CPL_KEY_W)) {
        cpl_cam_2D.pos.y -= speed * cpl_get_dt() * (1.0f / cpl_cam_2D.zoom);
    }
    if (cpl_is_key_down(CPL_KEY_S)) {
        cpl_cam_2D.pos.y += speed * cpl_get_dt() * (1.0f / cpl_cam_2D.zoom);
    }
    if (cpl_is_key_down(CPL_KEY_A)) {
        cpl_cam_2D.pos.x -= speed * cpl_get_dt() * (1.0f / cpl_cam_2D.zoom);
    }
    if (cpl_is_key_down(CPL_KEY_D)) {
        cpl_cam_2D.pos.x += speed * cpl_get_dt() * (1.0f / cpl_cam_2D.zoom);
    }
    if (cpl_is_key_down(CPL_KEY_ESCAPE)) {
        cpl_destroy_window();
    }

    if (cpl_is_key_down(CPL_KEY_H)) {
        cpl_cam_2D.zoom += 2 * cpl_get_dt();
    }
    if (cpl_is_key_down(CPL_KEY_N)) {
        cpl_cam_2D.zoom -= 2 * cpl_get_dt();
    }
    cpl_cam_2D.zoom = CPM_CLAMP(cpl_cam_2D.zoom, 0.1f, 10);
}

MAIN_PROG main(void) {
    cpl_init_window(800, 800, "Hello CPL", 33);
    cprng_rand_seed();

    f32 ambient_intensity = 0.01f;

    f32 start = cpl_get_time();
    f32 dt = 0.1f;

    texture pt;
    cpl_load_texture(&pt, "assets/images/ball.png", CPL_FILTER_LINEAR);

    particle_system ps;
    cpl_create_particle_system(&ps, VEC2F_INIT(400), UNLIMITED_PARTICLES, 100);

    while (!cpl_window_should_close()) {
        cpl_update();

        handle_movement();

        cpl_update_particle_system(&ps);
        if (cpl_get_time() >= start + dt) {
            for (u32 i = 0; i < cprng_rand_range(1, 3); i++) {
                cpl_add_particle(
                    &ps, PARTICLE(ps.pos, VEC2F_INIT(50),
                                  VEC2F(cprng_randf_range(-1, 1) * 300,
                                        cprng_randf_range(-1, 1) * 300),
                                  WHITE, cprng_randf_range(0, 2), 0, &pt));
            }
            start = cpl_get_time();
        }

        FOREACH_VEC(particle, vec_particle, p, &ps.particles) {
            p->color.a = 255 * (1 - (p->cur_life_time / p->life_time));
        }

        cpl_clear_background(BLACK);

        cpl_begin_draw(CPL_SHAPE_2D_LIT, true);

        cpl_set_ambient_light_2D(ambient_intensity);

        point_light_2D light[] = {
            {.pos = cpl_get_screen_to_world_2D(cpl_get_mouse_pos()),
             .intensity = 1,
             .color = WHITE,
             .radius = 500},
        };

        cpl_add_point_lights_2D(light, sizeof(light) / sizeof(point_light_2D));

        cpl_draw_rect(cpl_cam_2D.pos, cpl_get_screen_size(), WHITE, 0);

        cpl_draw_rect(VEC2F(300, 300), VEC2F_INIT(100), RED, 0);

        cpl_draw_rect(VEC2F(400, 400), VEC2F_INIT(300), BLUE, 0);

        cpl_begin_draw(CPL_TEXTURE_2D_LIT, true);

        cpl_draw_particles(&ps);

        cpl_begin_shadow_cast_2D();

        cpl_submit_rect_shadow(VEC2F(300, 300), VEC2F_INIT(100));
        cpl_submit_rect_shadow(VEC2F(400, 400), VEC2F_INIT(300));

        cpl_draw_shadows(light, sizeof(light) / sizeof(point_light_2D), 2000,
                        0.8f);

        cpl_end_shadow_cast_2D(ambient_intensity, 0.5f, RGB(0, 0, 0));

        cpl_end_frame();
    }
    cpl_close_window();
}
