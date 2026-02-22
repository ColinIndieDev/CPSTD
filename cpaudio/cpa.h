#pragma once

#include "../cpstd/cpbase.h"
#include "../cpstd/cpmath.h"
#include "../cpstd/cpvec.h"

#include <stdio.h>

#define CPA_WRITE_STR(f, s) fwrite((s), 1, sizeof(s) - 1, f)

void cpa_write_u16(FILE *f, u16 n) { fwrite(&n, sizeof(u16), 1, f); }

void cpa_write_u32(FILE *f, u32 n) { fwrite(&n, sizeof(u32), 1, f); }

FILE *cpa_create_wav(char *name) { return fopen(name, "wb"); }

typedef struct {
    f32 freq;
    f32 dur;
} note;

VEC_DEF(note, vec_note);

void cpa_fill_wav(FILE *f, vec_note *notes) {
    f32 dur = 0;

    for (size_t i = 0; i < notes->size; i++) {
        dur += notes->data[i].dur;
    }

    int freq = 44100;
    u32 sample_cnt = (u32)dur * freq;
    u32 defaultFlagSize = 44;
    u32 wavSize = (sample_cnt * sizeof(u16)) + 44;

    CPA_WRITE_STR(f, "RIFF");
    cpa_write_u32(f, wavSize - 8);
    CPA_WRITE_STR(f, "WAVE");

    CPA_WRITE_STR(f, "fmt ");
    cpa_write_u32(f, 16);
    cpa_write_u16(f, 1);
    cpa_write_u16(f, 1);
    cpa_write_u32(f, freq);
    cpa_write_u32(f, freq * sizeof(u16));
    cpa_write_u16(f, sizeof(u16));
    cpa_write_u16(f, sizeof(u16) * 8);

    CPA_WRITE_STR(f, "data");
    cpa_write_u32(f, sample_cnt * sizeof(u16));

    u32 cur_note = 0;
    f32 cur_note_start = 0.0f;

    for (int i = 0; i < sample_cnt; i++) {
        f32 t = (f32)i / (f32)freq;

        f32 y = 0.0f;
        if (cur_note < notes->size) {
            y = 0.25f * cpm_sinf(t * notes->data[cur_note].freq * 2.0f * CPM_PI);

            if (t > cur_note_start + notes->data[cur_note].dur) {
                cur_note++;
                cur_note_start = t;
            }
        }
        i16 sample = (i16)(y * CPM_I16_MAX);

        cpa_write_u16(f, sample);
    }
    printf("Your masterpiece is ready!\n");
}
