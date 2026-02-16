#include "../cpaudio/cpa.h"
#include "../cplterminal/cplt.h"
#include "../cpstd/cparr.h"
#include "../cpstd/cprng.h"

void DoCPA();
void DoCPLT();
void DoCPRNG();

int main() { DoCPRNG(); }

void DoCPA() {
    vec_note notes;

    vec_note_reserve(&notes, 16);

    // Note : Freq (Hz), Duration (s)
    vec_note_push_back(&notes, (note){392, 60.0f / 76.0f});
    vec_note_push_back(&notes, (note){440, 60.0f / 76.0f});
    vec_note_push_back(&notes, (note){294, 60.0f / 144.0f});
    vec_note_push_back(&notes, (note){440, 60.0f / 76.0f});
    vec_note_push_back(&notes, (note){494, 60.0f / 76.0f});

    FILE *f = cpa_create_wav("music.wav");

    cpa_fill_wav(f, &notes);

    fclose(f);
}

void DoCPLT() {
    cplt_init(75, 50);

    vec2f player = {(float)width / 2.0f, (float)height};
    float vel = 0.0f;

    bool start = false;

    while (running) {
        cplt_calc_fps();
        cplt_calc_dt();
        UpdateInput();

        float gravity = 40.0f;
        float jumpForce = 20.0f;
        if (cplt_is_key_pressed('w')) {
            if (!start) {
                start = true;
            }
            vel = -jumpForce;
        }

        if (start) {
            vel += gravity * dt;
        }
        player.y += vel * dt;

        cplt_clear_bg(0, 0, 0);
        cplt_clear(' ', 255, 255, 255);
        cplt_draw_pixel(4, 2, "#", 0, 0, 0);
        cplt_draw_rect(-1, 5, 4, 4, "#", 255, 255, 255);
        cplt_draw_circle((int)player.x, (int)player.y, 3, "#", 0, 255, 0);

        char fpsStr[25];
        snprintf(fpsStr, sizeof(fpsStr), "%s%d", "FPS: ", fps);
        cplt_draw_text(0, 0, fpsStr, 255, 0, 255);

        char posStr[25];
        snprintf(posStr, sizeof(posStr), "%s%f", "Y: ", player.y);
        cplt_draw_text(0, 1, posStr, 255, 0, 255);

        cplt_render();
    }
}

ARR_DEF(f32, arr_f32)

void DoCPRNG() {
    cprng_rand_seed();

    arr_f32 arr;
    arr_f32_init(&arr, 10);

    for (u32 i = 0; i < 10; i++) {
        *arr_f32_at(&arr, i) = cpm_modf(cprng_randf(), 10);
    }

    for (u32 i = 0; i < 10; i++) {
        printf("%f\n", *arr_f32_at(&arr, i));
    }

    arr_f32_destroy(&arr);
}
