#include "../cpaudio/cpa.h"

void DoCPA();

int main() {
   DoCPA(); 
}

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
