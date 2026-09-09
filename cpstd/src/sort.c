#include "../include/cpstd/sort.h"

#include <stdlib.h>
#include <string.h>

#define INSERTION_THRESHOLD 16

static int log2_floor(int n) {
    int result = 0;
    while (n > 1) {
        n >>= 1;
        result++;
    }
    return result;
}

void insertion_sort(void *a, size_t s, size_t n, int (*cmp)(const void *, const void *)) {
    for (size_t i = 1; i < n; ++i) {
        char *key = malloc(s);
        memcpy(key, a + i * s, s);
        int j = (int)i - 1;
        while (j >= 0 && cmp(a + j * s, key) > 0) {
            memcpy(a + (j + 1) * s, a + j * s, s);
            j--;
        }
        memcpy(a + (j + 1) * s, key, s);
        free(key);
    }
}

void swap(void *x, void *y, size_t s) {
    char *tmp = malloc(s);
    memcpy(tmp, x, s);
    memcpy(x, y, s);
    memcpy(y, tmp, s);
    free(tmp);
}

void heapify(void *a, size_t s, size_t n, int i, int (*cmp)(const void *, const void *)) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && cmp(a + l * s, a + largest * s) > 0) {
        largest = l;
    }
    if (r < n && cmp(a + r * s, a + largest * s) > 0) {
        largest = r;
    }
    if (largest != i) {
        swap(a + i * s, a + largest * s, s);
        heapify(a, s, n, largest, cmp);
    }
}

void heap_sort(void *a, size_t s, size_t n, int (*cmp)(const void *, const void *)) {
    for (int i = (int)n / 2; i > 0; --i) {
        heapify(a, s, n, i - 1, cmp);
    }
    for (int i = (int)n; i > 1; --i) {
       swap(a, a + (i - 1) * s, s);
       heapify(a, s, i - 1, 0, cmp);
    }
}

int partition(void *a, size_t s, int low, int high, int (*cmp)(const void *, const void *)) {
    char *pivot = malloc(s);
    memcpy(pivot, a + high * s, s);
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (cmp(a + j * s, pivot) <= 0) {
            i++;
            swap(a + i * s, a + j * s, s);
        }
    }
    swap(a + (i + 1) * s, a + high * s, s);
    free(pivot);
    return i + 1;
}

void quick_sort_recursive(void *a, size_t s, int low, int high, int (*cmp)(const void *, const void *)) {
    if (low < high) {
        int pi = partition(a, s, low, high, cmp);
        quick_sort_recursive(a, s, low, pi - 1, cmp);
        quick_sort_recursive(a, s, pi + 1, high, cmp);
    }
}

void quick_sort(void *a, size_t s, size_t n, int (*cmp)(const void *, const void *)) {
    int low = 0;
    int high = n - 1;
    if (low < high) {
        int pi = partition(a, s, low, high, cmp); 
        quick_sort_recursive(a, s, low, pi - 1, cmp);
        quick_sort_recursive(a, s, pi + 1, high, cmp);
    }
}

void introsort(void *a, size_t s, int low, int high, int depth_limit, int (*cmp)(const void *, const void *)) {
    int n = high - low + 1;
    if (n <= INSERTION_THRESHOLD) {
        insertion_sort(a + low * s, s, n, cmp);
        return;
    }
    if (depth_limit == 0) {
        heap_sort(a + low * s, s, n, cmp);
    }
    int pivot = partition(a, s, low, high, cmp);
    introsort(a, s, low, pivot - 1, depth_limit - 1, cmp);
    introsort(a, s, pivot + 1, high, depth_limit - 1, cmp);
}

void sort(void *a, size_t s, size_t n, int (*cmp)(const void *, const void *)) {
    if (n <= 1) {
        return;
    }
    int depth_limit = 2 * log2_floor(n);
    introsort(a, s, 0, n - 1, depth_limit, cmp);
}
