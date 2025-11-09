
#ifndef BMP_HEADER
#define BMP_HEADER 1

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

#define BMP_R 2
#define BMP_G 1
#define BMP_B 0

typedef int32_t int32;
typedef int16_t int16;
typedef unsigned char byte;

struct _bitmap_image
{
    byte *pixels;
    int32 width;
    int32 height;
    int32 bytes_per_pixel;
};

typedef struct _bitmap_image bitmap_image;

bitmap_image *NewBMPImage(int32 width, int32 height, int32 bytes_per_pixel);
bitmap_image *ReadBMPImage(const char *fileName);
void WriteBMPImage(const char *fileName, bitmap_image *img);
void DelBMPImage(bitmap_image *img);

bitmap_image *make_test_example();

void copy_scale(bitmap_image *input, bitmap_image *output);
void bicubic_interpolation(bitmap_image *input, bitmap_image *output);
void bicubic_hermite_interpolation(bitmap_image *input, bitmap_image *output);

#endif