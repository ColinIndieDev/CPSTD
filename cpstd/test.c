#include "cpbase.h"

#define INSERTION_THRESHOLD 16

static i32 log2_floor(i32 n) {
    i32 result = 0;
    while (n > 1) {
        n >>= 1;
        result++;
    }
    return result;
}

void integers_insertion_sort(i32*a, u32 size) {
    for (u32 i = 1; i < size; i++) {
        i32 key = a[i];
        i32 j = (i32)i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j -= 1;
        }
        a[j + 1] = key;
    }
}
void integers_swap(i32*x, i32*y) {
    i32 tmp = *x;
    *x = *y;
    *y = tmp;
}
void integers_heapify(i32*a, i32 n, i32 i) {
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
        integers_swap(&a[i], &a[largest]);
        integers_heapify(a, n, largest);
    }
}
void integers_heap_sort(i32*a, u32 size) {
    for (i32 i = ((i32)size / 2) - 1; i >= 0; i--) {
        integers_heapify(a, (i32)size, i);
    }
    for (i32 i = (i32)size - 1; i > 0; i--) {
        integers_swap(&a[0], &a[i]);
        integers_heapify(a, i, 0);
    }
}
i32 partition(i32*a, i32 low, i32 high) {
    i32 pivot = a[high];
    i32 i = low - 1;
    for (i32 j = low; j < high; j++) {
        if (a[j] <= pivot) {
            i++;
            integers_swap(&a[i], &a[j]);
        }
    }
    integers_swap(&a[i + 1], &a[high]);
    return i + 1;
}
void integers_quick_sort(i32*a, i32 low, i32 high) {
    if (low < high) {
        i32 pi = partition(a, low, high);
        integers_quick_sort(a, low, pi - 1);
        integers_quick_sort(a, pi + 1, high);
    }
}
void integers_introsort(i32*a, i32 low, i32 high, i32 depth_limit) {
    i32 size = high - low + 1;
    if (size <= INSERTION_THRESHOLD) {
        integers_insertion_sort(a + low, size);
        return;
    }
    if (depth_limit == 0) {
        integers_heap_sort(a + low, size);
        return;
    }
    i32 pivot = partition(a, low, high);
    integers_introsort(a, low, pivot - 1, depth_limit - 1);
    integers_introsort(a, pivot + 1, high, depth_limit - 1);
}
void integers_sort(i32*a, i32 size) {
    if (size <= 1) {
        return;
    }
    i32 depth_limit = 2 * log2_floor(size);
    integers_introsort(a, 0, size - 1, depth_limit);
}
