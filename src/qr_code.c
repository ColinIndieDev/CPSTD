#define CPL_IMPLEMENTATION
#include "../cplibrary/cpl.h"

#define VERSION 3
#define SIZE (17 + (4 * VERSION))

#if VERSION == 4
#define DATA_BYTES 64
#define EC_BYTES 36
#define TOTAL_CW (DATA_BYTES + EC_BYTES)
#elif VERSION == 2
#define DATA_BYTES 28
#define EC_BYTES 16
#define TOTAL_CW (DATA_BYTES + EC_BYTES)
#elif VERSION == 3
#define DATA_BYTES 44
#define EC_BYTES 26
#define TOTAL_CW (DATA_BYTES + EC_BYTES)
#endif

#define QUIET 2

u32 pixel_size = 25;

typedef enum {
    DATA_FORMAT_NUMERIC = 0b0001,
    DATA_FORMAT_ALPHANUMERIC = 0b0010,
    DATA_FORMAT_BINARY = 0b0100,
} data_format;

typedef enum {
    MASK_0,
    MASK_1,
    MASK_2,
    MASK_3,
    MASK_4,
    MASK_5,
    MASK_6,
    MASK_7,
} mask;

u32 qr_code[SIZE][SIZE] = {0};

// {{{ Fixed pattern

void write_position_detection_patterns() {
    qr_code[0][0] = 1;
    qr_code[0][1] = 1;
    qr_code[0][2] = 1;
    qr_code[0][3] = 1;
    qr_code[0][4] = 1;
    qr_code[0][5] = 1;
    qr_code[0][6] = 1;
    qr_code[1][0] = 1;
    qr_code[2][0] = 1;
    qr_code[3][0] = 1;
    qr_code[4][0] = 1;
    qr_code[5][0] = 1;
    qr_code[6][0] = 1;
    qr_code[6][1] = 1;
    qr_code[6][2] = 1;
    qr_code[6][3] = 1;
    qr_code[6][4] = 1;
    qr_code[6][5] = 1;
    qr_code[6][6] = 1;
    qr_code[1][6] = 1;
    qr_code[2][6] = 1;
    qr_code[3][6] = 1;
    qr_code[4][6] = 1;
    qr_code[5][6] = 1;
    qr_code[2][2] = 1;
    qr_code[2][3] = 1;
    qr_code[2][4] = 1;
    qr_code[3][2] = 1;
    qr_code[3][3] = 1;
    qr_code[3][4] = 1;
    qr_code[4][2] = 1;
    qr_code[4][3] = 1;
    qr_code[4][4] = 1;

    qr_code[(SIZE - 7) + 0][0] = 1;
    qr_code[(SIZE - 7) + 0][1] = 1;
    qr_code[(SIZE - 7) + 0][2] = 1;
    qr_code[(SIZE - 7) + 0][3] = 1;
    qr_code[(SIZE - 7) + 0][4] = 1;
    qr_code[(SIZE - 7) + 0][5] = 1;
    qr_code[(SIZE - 7) + 0][6] = 1;
    qr_code[(SIZE - 7) + 1][0] = 1;
    qr_code[(SIZE - 7) + 2][0] = 1;
    qr_code[(SIZE - 7) + 3][0] = 1;
    qr_code[(SIZE - 7) + 4][0] = 1;
    qr_code[(SIZE - 7) + 5][0] = 1;
    qr_code[(SIZE - 7) + 6][0] = 1;
    qr_code[(SIZE - 7) + 6][1] = 1;
    qr_code[(SIZE - 7) + 6][2] = 1;
    qr_code[(SIZE - 7) + 6][3] = 1;
    qr_code[(SIZE - 7) + 6][4] = 1;
    qr_code[(SIZE - 7) + 6][5] = 1;
    qr_code[(SIZE - 7) + 6][6] = 1;
    qr_code[(SIZE - 7) + 1][6] = 1;
    qr_code[(SIZE - 7) + 2][6] = 1;
    qr_code[(SIZE - 7) + 3][6] = 1;
    qr_code[(SIZE - 7) + 4][6] = 1;
    qr_code[(SIZE - 7) + 5][6] = 1;
    qr_code[(SIZE - 7) + 2][2] = 1;
    qr_code[(SIZE - 7) + 2][3] = 1;
    qr_code[(SIZE - 7) + 2][4] = 1;
    qr_code[(SIZE - 7) + 3][2] = 1;
    qr_code[(SIZE - 7) + 3][3] = 1;
    qr_code[(SIZE - 7) + 3][4] = 1;
    qr_code[(SIZE - 7) + 4][2] = 1;
    qr_code[(SIZE - 7) + 4][3] = 1;
    qr_code[(SIZE - 7) + 4][4] = 1;

    qr_code[0][(SIZE - 7) + 0] = 1;
    qr_code[0][(SIZE - 7) + 1] = 1;
    qr_code[0][(SIZE - 7) + 2] = 1;
    qr_code[0][(SIZE - 7) + 3] = 1;
    qr_code[0][(SIZE - 7) + 4] = 1;
    qr_code[0][(SIZE - 7) + 5] = 1;
    qr_code[0][(SIZE - 7) + 6] = 1;
    qr_code[1][(SIZE - 7) + 0] = 1;
    qr_code[2][(SIZE - 7) + 0] = 1;
    qr_code[3][(SIZE - 7) + 0] = 1;
    qr_code[4][(SIZE - 7) + 0] = 1;
    qr_code[5][(SIZE - 7) + 0] = 1;
    qr_code[6][(SIZE - 7) + 0] = 1;
    qr_code[6][(SIZE - 7) + 1] = 1;
    qr_code[6][(SIZE - 7) + 2] = 1;
    qr_code[6][(SIZE - 7) + 3] = 1;
    qr_code[6][(SIZE - 7) + 4] = 1;
    qr_code[6][(SIZE - 7) + 5] = 1;
    qr_code[6][(SIZE - 7) + 6] = 1;
    qr_code[1][(SIZE - 7) + 6] = 1;
    qr_code[2][(SIZE - 7) + 6] = 1;
    qr_code[3][(SIZE - 7) + 6] = 1;
    qr_code[4][(SIZE - 7) + 6] = 1;
    qr_code[5][(SIZE - 7) + 6] = 1;
    qr_code[2][(SIZE - 7) + 2] = 1;
    qr_code[2][(SIZE - 7) + 3] = 1;
    qr_code[2][(SIZE - 7) + 4] = 1;
    qr_code[3][(SIZE - 7) + 2] = 1;
    qr_code[3][(SIZE - 7) + 3] = 1;
    qr_code[3][(SIZE - 7) + 4] = 1;
    qr_code[4][(SIZE - 7) + 2] = 1;
    qr_code[4][(SIZE - 7) + 3] = 1;
    qr_code[4][(SIZE - 7) + 4] = 1;
}

void draw_alignment(u32 r, u32 c) {
    for (i32 dr = -2; dr <= 2; dr++) {
        for (i32 dc = -2; dc <= 2; dc++) {
            if (dr == -2 || dr == 2 || dc == -2 || dc == 2 ||
                (dr == 0 && dc == 0)) {
                qr_code[r + dr][c + dc] = 1;
            } else {
                qr_code[r + dr][c + dc] = 0;
            }
        }
    }
}

void write_aligment_pattern() { draw_alignment(SIZE - 7, SIZE - 7); }

void write_timing() {
    for (u32 i = 8; i < SIZE - 7; i += 2) {
        qr_code[6][i] = 1;
        qr_code[i][6] = 1;
    }
}

// }}}

// {{{ Bitstream

typedef struct {
    u8 data[400];
    u32 len;
} bit_stream;

void bs_append(bit_stream *bs, u32 value, i32 n_bits) {
    for (i32 i = n_bits - 1; i >= 0; i--) {
        u32 bit = (value >> i) & 1;
        bs->data[bs->len / 8] |= (bit << (7 - (bs->len % 8)));
        bs->len++;
    }
}

void encode_byte_mode(char *input, bit_stream *bs, data_format fmt) {
    u32 len = strlen(input);
    memset(bs, 0, sizeof(*bs));

    bs_append(bs, fmt, 4);
    bs_append(bs, len, 8);

    for (u32 i = 0; i < len; i++) {
        bs_append(bs, (u8)input[i], 8);
    }

    bs_append(bs, 0b0000, 4);

    while (bs->len % 8 != 0) {
        bs_append(bs, 0, 1);
    }

    u8 pad_bytes[] = {0xEC, 0x11};
    i32 i = 0;

    while (bs->len / 8 < DATA_BYTES) {
        bs_append(bs, pad_bytes[i++ % 2], 8);
    }
}

// }}}

// {{{ Error correction (Polynomial)

static u8 gf_exp[512], gf_log[256];

void gf_init() {
    i32 x = 1;
    for (u32 i = 0; i < 255; i++) {
        gf_exp[i] = x;
        gf_log[x] = i;
        x <<= 1;
        if (x & 0x100) {
            x ^= 0x11D;
        }
    }
    for (u32 i = 255; i < 512; i++) {
        gf_exp[i] = gf_exp[i - 255];
    }
}

u8 gf_mul(u8 a, u8 b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return gf_exp[gf_log[a] + gf_log[b]];
}

static const u8 gen_poly_16[16] = {59,  13, 104, 189, 68, 209, 30, 8,
                                   163, 65, 41,  229, 98, 50,  36, 59};

static const u8 gen_poly_18[18] = {215, 234, 158, 94,  184, 97, 118, 170, 79,
                                   187, 152, 148, 252, 179, 5,  98,  96,  153};

static const u8 gen_poly_26[26] = {246, 51,  183, 4,   136, 98,  199, 152, 77,
                                   56,  206, 24,  145, 40,  209, 117, 233, 42,
                                   135, 68,  70,  144, 146, 77,  43,  94};

void reed_solomon_16(u8 *data, u32 data_len, u8 *ec) {
    u8 buf[200] = {0};
    memcpy(buf, data, data_len);

    for (u32 i = 0; i < data_len; i++) {
        u8 coef = buf[i];
        if (coef == 0) {
            continue;
        }
        for (u32 j = 0; j < 16; j++) {
            buf[i + 1 + j] ^= gf_mul(gen_poly_16[j], coef);
        }
    }
    memcpy(ec, buf + data_len, 16);
}

void reed_solomon_18(u8 *data, u32 data_len, u8 *ec) {
    u8 buf[200] = {0};
    memcpy(buf, data, data_len);

    for (u32 i = 0; i < data_len; i++) {
        u8 coef = buf[i];
        if (coef == 0) {
            continue;
        }
        for (u32 j = 0; j < 18; j++) {
            buf[i + 1 + j] ^= gf_mul(gen_poly_18[j], coef);
        }
    }
    memcpy(ec, buf + data_len, 18);
}

void reed_solomon_26(u8 *data, u32 data_len, u8 *ec) {
    u8 buf[200] = {0};
    memcpy(buf, data, data_len);

    for (u32 i = 0; i < data_len; i++) {
        u8 coef = buf[i];
        if (coef == 0) {
            continue;
        }
        for (u32 j = 0; j < 26; j++) {
            buf[i + 1 + j] ^= gf_mul(gen_poly_26[j], coef);
        }
    }

    memcpy(ec, buf + data_len, 26);
}

// }}}

// {{{ Data, format & mask

b8 function_module(u32 r, u32 c) {
    if (r <= 8 && c <= 8) {
        return true;
    }
    if (r <= 8 && c >= SIZE - 8) {
        return true;
    }
    if (r >= SIZE - 8 && c <= 8) {
        return true;
    }
    if (r == 6 || c == 6) {
        return true;
    }
    u32 align = SIZE - 7;
    if (r >= align - 2 && r <= align + 2 && c >= align - 2 && c <= align + 2) {
        return true;
    }
    if (r == 8 && (c <= 8 || c >= SIZE - 8)) {
        return true;
    }
    if (c == 8 && (r <= 8 || r >= SIZE - 8)) {
        return true;
    }
    if (r == (4 * VERSION) + 9 && c == 8) {
        return true;
    }
    return false;
}

void place_data(const u8 *code_words, u32 cw_len) {
    i32 bit_idx = 0;
    u32 total_bits = cw_len * 8;
    i32 col = SIZE - 1;
    while (col > 0) {
        if (col == 6) {
            col--;
            continue;
        }
        i32 strip = (col > 6) ? (SIZE - 1 - col) / 2 : (SIZE - 1 - col - 1) / 2;
        i32 go_up = (strip % 2 == 0);
        for (i32 i = 0; i < SIZE; i++) {
            i32 row = go_up ? (SIZE - 1 - i) : i;
            if (!function_module(row, col)) {
                if (bit_idx < total_bits) {
                    i32 byte = bit_idx / 8;
                    i32 bit = 7 - (bit_idx % 8);
                    qr_code[row][col] = (code_words[byte] >> bit) & 1;
                } else {
                    qr_code[row][col] = 0;
                }
                bit_idx++;
            }
            if (!function_module(row, col - 1)) {
                if (bit_idx < total_bits) {
                    i32 byte = bit_idx / 8;
                    i32 bit = 7 - (bit_idx % 8);
                    qr_code[row][col - 1] = (code_words[byte] >> bit) & 1;
                } else {
                    qr_code[row][col - 1] = 0;
                }
                bit_idx++;
            }
        }
        col -= 2;
    }
}

void flip_bit(u32 r, u32 c) { qr_code[r][c] = qr_code[r][c] == 0 ? 1 : 0; }

void apply_mask(mask m) {
    for (u32 r = 0; r < SIZE; r++) {
        for (u32 c = 0; c < SIZE; c++) {
            if (!function_module(r, c)) {
                switch (m) {
                case MASK_0:
                    if ((r + c) % 2 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_1:
                    if (r % 2 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_2:
                    if (c % 3 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_3:
                    if ((r + c) % 3 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_4:
                    if ((r / 2 + c / 3) % 2 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_5:
                    if (((r * c) % 2) + ((r * c) % 3) == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_6:
                    if (((r * c) % 2 + (r * c) % 3) % 2 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                case MASK_7:
                    if (((r + c) % 2 + (r * c) % 3) % 2 == 0) {
                        flip_bit(r, c);
                    }
                    break;
                default:
                    break;
                }
            }
        }
    }
}

static const u16 FMT_M[8] = {0x5412, 0x5125, 0x5E7C, 0x5B4B,
                             0x45F9, 0x40CE, 0x4F97, 0x4AA0};
static const u16 FMT_L[8] = {0x77C4, 0x72F3, 0x7DAA, 0x789D,
                             0x662F, 0x6318, 0x6C41, 0x6976};
static const u16 FMT_Q[8] = {0x1EB5, 0x1B82, 0x14DB, 0x11EC,
                             0x0F5E, 0x0A69, 0x0530, 0x0007};
static const u16 FMT_H[8] = {0x26A5, 0x2392, 0x2CCB, 0x29FC,
                             0x374E, 0x3279, 0x3D20, 0x3817};

void write_fmt(mask m) {
    u16 fmt = FMT_M[m];
    i32 col_seq[] = {0, 1, 2, 3, 4, 5, 7, 8};
    for (u32 i = 0; i < 8; i++) {
        i32 bit = (fmt >> (14 - i)) & 1;
        qr_code[8][col_seq[i]] = bit;
    }
    i32 row_seq[] = {7, 5, 4, 3, 2, 1, 0};
    for (u32 i = 0; i < 7; i++) {
        i32 bit = (fmt >> (6 - i)) & 1;
        qr_code[row_seq[i]][8] = bit;
    }
    for (u32 i = 0; i < 8; i++) {
        qr_code[8][SIZE - 8 + i] = (fmt >> i) & 1;
    }
    for (u32 i = 0; i < 8; i++) {
        qr_code[SIZE - 1 - i][8] = (fmt >> i) & 1;
    }
    qr_code[(4 * VERSION) + 9][8] = 1;
}

// }}}

// {{{ QR Code

void qr_encode(char *input, data_format fmt) {
    gf_init();

    bit_stream bs;
    encode_byte_mode(input, &bs, fmt);

#if VERSION == 4
    u8 block1[DATA_BYTES / 2];
    u8 block2[DATA_BYTES / 2];

    memcpy(block1, bs.data, DATA_BYTES / 2);
    memcpy(block2, bs.data + (DATA_BYTES / 2), DATA_BYTES / 2);

    u8 ec1[EC_BYTES / 2];
    u8 ec2[EC_BYTES / 2];

    reed_solomon_18(block1, DATA_BYTES / 2, ec1);
    reed_solomon_18(block2, DATA_BYTES / 2, ec2);

    u8 final_cw[TOTAL_CW];
    u32 idx = 0;

    for (u32 i = 0; i < DATA_BYTES / 2; i++) {
        final_cw[idx++] = block1[i];
        final_cw[idx++] = block2[i];
    }

    for (u32 i = 0; i < EC_BYTES / 2; i++) {
        final_cw[idx++] = ec1[i];
        final_cw[idx++] = ec2[i];
    }

    place_data(final_cw, TOTAL_CW);

#elif VERSION == 2
    u8 ec[EC_BYTES];
    reed_solomon_16(bs.data, DATA_BYTES, ec);

    u8 final_cw[TOTAL_CW];
    memcpy(final_cw, bs.data, DATA_BYTES);
    memcpy(final_cw + DATA_BYTES, ec, EC_BYTES);

    place_data(final_cw, TOTAL_CW);
#elif VERSION == 3
    u8 ec[EC_BYTES];
    reed_solomon_26(bs.data, DATA_BYTES, ec);

    u8 final_cw[TOTAL_CW];
    memcpy(final_cw, bs.data, DATA_BYTES);
    memcpy(final_cw + DATA_BYTES, ec, EC_BYTES);

    place_data(final_cw, TOTAL_CW);
#endif
}

void create_qr_code(char *input, data_format fmt, mask mask) {
#if VERSION == 4 || VERSION == 3 || VERSION == 2
    write_position_detection_patterns();
    write_aligment_pattern();
    write_timing();
    qr_encode(input, fmt);
    apply_mask(mask);
    write_fmt(mask);
#else
    fprintf(stderr, "Version not supported!");
    exit(-1);
#endif
}

// }}}

MAIN_PROG main(void) {
    cpl_init_window(800, 800, "QR Code", OPENGL_VER_3_3);

    char link[100];
    printf("-------------------------------------------------------------------\n");
    printf("Insert link (Note that these versions may be restricted by data)\n");
    printf("-------------------------------------------------------------------\n");
    printf("Link >> ");
    fgets(link, sizeof(link), stdin);

    // If using it for videos on yt, please avoid https or www to save data space:
    // youtube.com/watch?v=xvFZjo5PgG0

    create_qr_code(link, DATA_FORMAT_BINARY, MASK_5);

    while (!cpl_window_should_close()) {
        cpl_update();

        if (cpl_is_key_pressed(CPL_KEY_ESCAPE)) {
            cpl_destroy_window();
        }

        pixel_size = CPM_MIN(cpl_get_screen_width(), cpl_get_screen_height()) / (SIZE + (QUIET * 2));

        cpl_clear_background(WHITE);

        cpl_begin_draw(CPL_SHAPE_2D_UNLIT, false);

        for (u32 r = 0; r < SIZE; r++) {
            for (u32 c = 0; c < SIZE; c++) {
                if (qr_code[r][c] == 1) {
                    cpl_draw_rect(
                        VEC2F((QUIET * pixel_size) + (c * pixel_size),
                              (QUIET * pixel_size) + (r * pixel_size)),
                        VEC2F_INIT(pixel_size), BLACK, 0);
                }
            }
        }

        cpl_end_frame();
    }
    cpl_close_window();
}
