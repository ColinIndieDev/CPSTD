#pragma once

#define PAIR_DEF(t1, t2, name)                                                 \
    typedef struct {                                                           \
        t1 first;                                                              \
        t2 second;                                                             \
    } name;
