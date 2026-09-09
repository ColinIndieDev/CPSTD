#include "../include/cpstd/bitarr.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

uint8_t *bit_arr_init(size_t bits) {
    assert(bits > 0);
    size_t bytes = (bits + 7) / 8;
    bit_header_t *header = malloc(sizeof(bit_header_t) + bytes);
    header->bits = bits;
    uint8_t *arr = (uint8_t *)(header + 1);
    memset(arr, 0, bytes);
    return arr;
}

void bit_arr_destroy(uint8_t *arr) {
    assert(arr);
    free(bit_header(arr));
}
void bit_arr_set(uint8_t *arr, size_t i) {
    assert(arr);
    assert(i >= 0 && i < bit_header(arr)->bits);
    size_t byte = i >> 3;
    size_t bit = i & 7;
    arr[byte] |= (1 << bit);
}
void bit_arr_clear(uint8_t *arr, size_t i) {
    assert(arr);
    assert(i >= 0 && i < bit_header(arr)->bits);
    size_t byte = i >> 3;
    size_t bit = i & 7;
    arr[byte] &= ~(1 << bit);
}

uint8_t bit_arr_get(const uint8_t *arr, size_t i) {
    assert(arr);
    assert(i >= 0 && i < bit_header(arr)->bits);
    size_t byte = i >> 3;
    size_t bit = i & 7;
    return (arr[byte] >> bit) & 1;
}
