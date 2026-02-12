
#include "../cpstd/cpbase.h"
#include "../cpstd/cppair.h"
#include "../cpstd/cpstr.h"
#include "../cpstd/cpvec.h"
#include "../cpstd/cpmath.h"

VEC_DEF(i32, vec_i32)
VEC_DEF(vec_i32 *, vec_vec_i32);

int main() {
    vec_vec_i32 vec;
    vec_vec_i32_reserve(&vec, 5);

    for (int i = 0; i < 5; i++) {
        vec_i32 *v = malloc(sizeof(vec_i32));
        vec_i32_init(v, 5, i);

        vec_vec_i32_push_back(&vec, v);
    }

    vec_i32_pop_back(vec.data[0]);
    vec.data[1]->data[0] = 67;

    for (int i = 0; i < vec.size; i++) {
        for (int j = 0; j < vec.data[i]->size; j++) {
            int val = vec.data[i]->data[j];
            printf("%d", val);

            if (j < vec.data[i]->size - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }

    mat2f matrix;

    mat2f_init(&matrix, 3, 7, 25);

    mat2f_print(&matrix);
}
