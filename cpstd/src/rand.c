#include "../include/cpstd/rand.h"

#include <stdint.h>
#include <time.h>
#include <math.h>

static pcg_state _rand_state = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

unsigned int pcg_randr(pcg_state *rng) {
    size_t oldState = rng->state;
    rng->state = (oldState * 6364136223846793005ULL) + rng->inc;
    unsigned int xorshifted = ((oldState >> 18u) ^ oldState) >> 27u;
    unsigned int rot = oldState >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void pcg_rand_seedr(pcg_state *rng) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    size_t seed = ((size_t)ts.tv_sec << 32) ^ ts.tv_nsec;

    rng->state = 0;
    rng->inc = (seed << 1u) | 1u;

    pcg_randr(rng);
    rng->state += seed;
    pcg_randr(rng);
}

void pcg_rand_seed() { pcg_rand_seedr(&_rand_state); }

float pcg_randfr(pcg_state *rng) {
    return (float)pcg_randr(rng) / (float)UINT32_MAX;
}

unsigned int pcg_rand() { return pcg_randr(&_rand_state); }

float pcg_randf_norm() { return pcg_randfr(&_rand_state); }

float pcg_randf() {
    return floorf((float)pcg_rand() / 100000.0f) + pcg_randf_norm();
}

float pcg_randf_range(float min, float max) {
    return min + (pcg_randf_norm() * (max - min));
}

int pcg_rand_range(int min, int max) {
    return min + (int)(pcg_rand() % (max - min + 1));
}

int pcg_rand_percent(float probability) {
    if (probability < 0.0f) {
        probability = 0.0f;
    }
    else if (probability > 1.0f) {
        probability = 1.0f;
    }

    return ((float)pcg_rand() / UINT32_MAX < probability);
}
