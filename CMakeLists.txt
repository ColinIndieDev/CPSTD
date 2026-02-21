cmake_minimum_required(VERSION 3.31)
project(Example)

# C
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#### Debug ####

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -O0 -fno-omit-frame-pointer)
    add_compile_options(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
        -O0
    )
    add_link_options(
        -fsanitize=address,undefined
    )
endif()

#### Dev ####

if (CMAKE_BUILD_TYPE STREQUAL "Dev")
    add_compile_options(-g -O2 -DNDEBUG)
endif()

#### Release ####

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-g -O3 -ffast-math -DNDEBUG)
endif()

#### Linking ####

# Is used here but normally never do 
# it name the files manually

file(GLOB_RECURSE SOURCE_FILES CONFIGURE DEPENDS cpstd/*.h cplterminal/*.h cpaudio/*.h)
message(STATUS "============================================")
message(STATUS "Included files")
message(STATUS "============================================")
message(STATUS "----- Files (.c & .h) -----")
foreach(f IN LISTS SOURCE_FILES)
    file(RELATIVE_PATH rel ${CMAKE_SOURCE_DIR} ${f})
    message(STATUS "  ${rel}")
endforeach()

add_executable(Example
    src/cpai_main.c
    ${SOURCE_FILES}
)

# math.h
# target_link_libraries(Example m)

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(Example PRIVATE OpenMP::OpenMP_C)
target_compile_options(Example PRIVATE -mavx2 -mfma)
