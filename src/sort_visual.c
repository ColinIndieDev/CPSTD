#define CPL_IMPLEMENTATION
#include "../cplibrary/cpl.h"

#include "../cpstd/cprng.h"

#define SIZE 200
#define MAX_VAL 1000

typedef enum {
    SORT_HEAP = 1,
    SORT_QUICK,
} sort_algorithms;

u32 unsorted[SIZE];
sort_algorithms sort;

f32 block_width = 1.0f;
f32 block_height = 0.0f;

f32 start = 0.0f;
f32 elapsed = 0.0f;

f32 dt = 0.01f;

audio beep;

void play_swap_sound(u32 val) {
    f32 t = (f32)val / MAX_VAL;
    beep.pitch = 0.5f + (t * 1.5f);
    beep.volume = 0.1f;

    cpl_audio_play_sound(&beep);
}

void sort_swap(u32 *x, u32 *y) {
    play_swap_sound(*x);
    u32 tmp = *x;
    *x = *y;
    *y = tmp;
}

// Heap
b8 heapified = false;
i32 idx = 0;
i32 idx_heapify = 0;

void sort_heapify(u32 *a, u32 n, i32 i) {
    i32 largest = i;
    i32 l = (2 * i) + 1;
    i32 r = (2 * i) + 2;
    if (l < n && a[l] > a[largest]) {
        largest = l;
    }
    if (r < n && a[r] > a[largest]) {
        largest = r;
    }
    if (largest != i) {
        sort_swap(&a[i], &a[largest]);
        sort_heapify(a, n, largest);
    }
}
void heapify() {
    if (cpl_get_time() > start + dt) {
        if (idx_heapify >= 0) {
            sort_heapify(unsorted, SIZE, idx_heapify);
            idx_heapify--;
            start = cpl_get_time();
        } else {
            heapified = true;
        }
    }
}
void heap_sort() {
    if (idx > 0 && cpl_get_time() > start + dt) {
        sort_swap(&unsorted[0], &unsorted[idx]);
        sort_heapify(unsorted, idx, 0);
        idx--;
        start = cpl_get_time();
    }
}

// Quick
typedef struct {
    i32 low;
    i32 high;
} range;
i32 i = 0;
i32 j = 0;
i32 top = -1;
range stack[SIZE];
range cur;
b8 partitioning = false;

void quick_sort() {
    if (!partitioning) {
        if (top >= 0) {
            cur = stack[top--];
            i = cur.low - 1;
            j = cur.low;
            partitioning = true;
        } else {
            return;
        }
    }
    i32 pivot = (i32)unsorted[cur.high];
    if (j < cur.high) {
        if (unsorted[j] <= pivot) {
            i++;
            sort_swap(&unsorted[i], &unsorted[j]);
        }
        j++;
    } else {
        sort_swap(&unsorted[i + 1], &unsorted[cur.high]);
        i32 pi = i + 1;
        if (pi - 1 > cur.low) {
            stack[++top] = (range){cur.low, pi - 1};
        }
        if (pi + 1 < cur.high) {
            stack[++top] = (range){pi + 1, cur.high};
        }
        partitioning = false;
    }
}

MAIN_PROG main(void) {
    cpl_init_window(800, 800, "Sort algorithms", 33);
    cpl_audio_init();
    cpl_enable_vsync(false);

    font f;
    cpl_create_font(&f, "assets/fonts/arial.ttf", "default", CPL_FILTER_LINEAR);

    beep = cpl_load_audio("assets/sounds/beep.mp3");

    cprng_rand_seed();

    printf("----------------------------------------------\n");
    printf("Please select one of the sort algorithms:\n");
    printf("[1] Heap Sort\n");
    printf("[2] Quick Sort\n");
    printf("----------------------------------------------\n");
    printf("(Enter number) >> ");

    scanf("%d", &sort);

    for (u32 i = 0; i < SIZE; i++) {
        unsorted[i] = cprng_rand_range(1, MAX_VAL);
    }

    if (sort == SORT_HEAP) {
        idx = SIZE - 1;
        idx_heapify = (SIZE / 2) - 1;
    } else if (sort == SORT_QUICK) {
        stack[++top] = (range){0, SIZE - 1};
    }

    start = cpl_get_time();

    while (!cpl_window_should_close()) {
        cpl_update();

        if (cpl_is_key_pressed(CPL_KEY_ESCAPE)) {
            cpl_destroy_window();
        }

        block_width = cpl_get_screen_width() / SIZE;
        block_height = cpl_get_screen_height() / MAX_VAL; 

        if (sort == SORT_HEAP) {
            if (heapified) {
                heap_sort();
            } else {
                heapify();
            }
        } else if (sort == SORT_QUICK) {
            if (cpl_get_time() > start + dt) {
                quick_sort();
                start = cpl_get_time();
            }
        }

        u32 sorted = 0;

        cpl_clear_background(BLACK);

        cpl_begin_draw(CPL_SHAPE_2D_UNLIT, false);

        for (i32 i = SIZE - 1; i >= 0; i--) {
            vec4f color = RED;
            if (i != 0 && unsorted[i - 1] <= unsorted[i]) {
                color = LIME_GREEN;
                sorted++;
            }
            if (i == 0 && unsorted[i] <= unsorted[i + 1]) {
                color = LIME_GREEN;
                sorted++;
            }

            cpl_draw_rect(
                VEC2F(i * block_width, cpl_get_screen_height() - (unsorted[i] * block_height)),
                VEC2F(block_width, unsorted[i] * block_height), color, 0);
        }

        cpl_begin_draw(CPL_TEXT, false);

        {
            char txt[100];
            snprintf(txt, 100, "Elements sorted: %d / %d", sorted, SIZE);
            cpl_draw_text_shadow(&f, txt,
                                 VEC2F(10, cpl_get_screen_height() - 60), 1,
                                 WHITE, VEC2F(4, 4), BLACK);
        }
        {
            char *name;
            char *best;
            char *avg;
            char *worst;
            char *space;

            switch (sort) {
            case SORT_HEAP:
                name = "Heap Sort";
                best = "Best: O(n log(n))";
                avg = "Avg: O(n log(n))";
                worst = "Worst: O(n log(n))";
                space = "Space: O(1)";
                break;
            case SORT_QUICK:
                name = "Quick Sort";
                best = "Best: O(n log(n))";
                avg = "Avg: O(n log(n))";
                worst = "Worst: O(n^2)";
                space = "Space: O(log(n))";
                break;
            }

            cpl_draw_text_shadow(&f, name, VEC2F(10, 10), 1.2f, WHITE,
                                 VEC2F(4, 4), BLACK);
            cpl_draw_text_shadow(&f, best, VEC2F(10, 75), 0.5f, WHITE,
                                 VEC2F(4, 4), BLACK);

            cpl_draw_text_shadow(&f, avg, VEC2F(10, 110), 0.5f, WHITE,
                                 VEC2F(4, 4), BLACK);
            cpl_draw_text_shadow(&f, worst, VEC2F(10, 145), 0.5f, WHITE,
                                 VEC2F(4, 4), BLACK);
            cpl_draw_text_shadow(&f, space, VEC2F(10, 180), 0.5f, WHITE,
                                 VEC2F(4, 4), BLACK);
        }

        cpl_end_frame();

        if (sorted >= SIZE) {
            for (u32 i = 0; i < SIZE; i++) {
                unsorted[i] = cprng_rand_range(1, MAX_VAL);
            }
            if (sort == SORT_HEAP) {
                heapified = false;
                idx = SIZE - 1;
                idx_heapify = (SIZE / 2) - 1;
            } else if (sort == SORT_QUICK) {
                i = 0;
                j = 0;
                top = -1;
                partitioning = false;
                stack[++top] = (range){0, SIZE - 1};
            }
            usleep(2 * 1000000);
        }
    }
    cpl_audio_close();
    cpl_close_window();
}
