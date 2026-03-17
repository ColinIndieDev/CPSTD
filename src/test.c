#include "../cpstd/cpbitarr.h"
#include "../cpstd/cphash.h"
#include "../cpstd/cplist.h"
#include "../cpstd/cpmemory.h"
#include "../cpstd/cpqueue.h"
#include "../cpstd/cparr.h"
#include "../cpstd/cpsort.h"

#include "../cpstd/cprng.h"

#include <stdio.h>
#include <time.h>

ARR_DEF(f32, arr_f32)
VEC_DEF(i32, vec_i32)
HASHMAP_DEF(u8, f32, map)
HASHSET_DEF(u8, set)
QUEUE_DEF(i32, queue)
PRIORITY_QUEUE_DEF(i32, priority_queue)
LINKED_LIST_DEF(const i8 *, ll_str)

SORT_ALGORITHM_DEF(i32, integers)

int main() {
    for (int i = 0; i < 5; i++) {
        int *val = cp_malloc(sizeof(int));

        *val = 1092;

        printf("%p: %d\n", val, *val);
        cp_free(val);
    }

    map m;
    map_init(&m, 3);

    map_put(&m, 'a', 197.0f);
    map_put(&m, 'z', 82.0f);
    map_put(&m, 'c', 996.198f);
    map_put(&m, 'b', 6.5f);
    map_put(&m, 'p', 8.278f);

    map_remove(&m, 'c');

    FOREACH_HM(map, entry, &m) {
        if (entry->state == HASH_OCCUPIED) {
            printf("%c: %f\n", entry->key, entry->value);
        }
    }

    map_destroy(&m);

    set s;
    set_init(&s, 3);

    set_put(&s, 'a');
    set_put(&s, 'z');
    set_put(&s, 'c');
    set_put(&s, 'b');
    set_put(&s, 'p');

    set_remove(&s, 'c');

    FOREACH_HS(set, entry, &s) {
        if (entry->state == HASH_OCCUPIED) {
            printf("%c\n", entry->key);
        }
    }

    for (int i = 'a'; i < ('z' + 1); i++) {
        if (set_contains(&s, i)) {
            printf("Found key: %c!\n", i);
        }
    }

    set_destroy(&s);

    priority_queue q;
    priority_queue_init(&q, 1);

    priority_queue_push(&q, 6, 6);
    priority_queue_push(&q, 11, 11);
    priority_queue_push(&q, -2, -2);
    priority_queue_push(&q, 7, 7);
    priority_queue_push(&q, 4, 4);

    printf("priority queue: ");
    while (!priority_queue_empty(&q)) {
        i32 val = 0;
        priority_queue_pop(&q, &val);
        printf("%d ", val);
    }
    printf("\n");

    priority_queue_destroy(&q);

    bit_arr ba;
    bit_arr_init(&ba, 10);

    bit_arr_set(&ba, 0);
    bit_arr_set(&ba, 3);
    bit_arr_set(&ba, 5);
    bit_arr_set(&ba, 7);
    bit_arr_set(&ba, 8);

    bit_arr_clear(&ba, 1);
    bit_arr_clear(&ba, 5);

    printf("bit array: ");
    for (u32 i = 0; i < ba.bits; i++) {
        printf("%d", bit_arr_get(&ba, i));
    }
    printf("\n");

    bit_arr_destroy(&ba);

    ll_str friends;
    ll_str_reserve(&friends, 10);
    ll_str_add(&friends, "Joe");
    ll_str_add(&friends, "Kathrina");
    ll_str_add(&friends, "Tom");
    ll_str_add(&friends, "Klaus");
    ll_str_add(&friends, "Hannah");
    ll_str_add(&friends, "Jimmy");

    ll_str_pop(&friends, 5);
    ll_str_pop(&friends, 4);

    ll_str_node *f = ll_str_get(&friends, 0);
    while (f != NULL) {
        printf("%s ", f->val);
        f = f->next;
    }
    printf("\n");

    ll_str_destroy(&friends);

    vec_i32 v1;
    vec_i32 v2;

    vec_i32_init(&v1, 10, 0);
    vec_i32_init(&v2, 0, 0);

    for (int i = 0; i < v1.size; i++) {
        *vec_i32_at(&v1, i) = i;
    }

    vec_i32_copy(&v1, &v2);

    FOREACH_VEC(int, vec_i32, it, &v2) { printf("%d", *it); }

    cprng_rand_seed();

    vec_i32 v;

    vec_i32_reserve(&v, 1000);
    for (u32 i = 0; i < 1000; i++) {
        vec_i32_push_back(&v, cprng_rand_range(0, 100000));
    }

    clock_t start = clock();
    integers_sort(v.data, (i32)v.size);

    f64 elapsed = (f64)(clock() - start) / CLOCKS_PER_SEC;

    printf("\nTook %.1fms!\n", elapsed * 1000000);
    FOREACH_VEC(i32, vec_i32, it, &v) { printf("%d ", *it); }
}
