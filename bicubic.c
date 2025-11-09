
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "stopwatch.h"
#include "BMP.h"

int main(int argc, char **argv)
{
    bitmap_image *img_input = NULL;
    bitmap_image *img_output = NULL;
    pstopwatch sw = NULL;
    char *filename = NULL;
    char *outname = "out";
    char *outfile = NULL;
    int32 scalefactor = 2;
    int32 length;

    sw = new_stopwatch();

    printf(BCYAN "--------------------------------------------------------------------------------\n");
    printf("  Magnifying image with bicubic interpolation:\n");
    printf("--------------------------------------------------------------------------------\n" NORMAL);

    // handle input file argument
    start_stopwatch(sw);
    if (argc >= 2)
    {
        filename = argv[1];
        printf("Reading image " BGREEN "\"%s" NORMAL "\"\n", filename);
        img_input = ReadBMPImage(filename);
    }
    else
    {
        printf("Creating test image\n");
        img_input = make_test_example();
    }
    assert(img_input != NULL);
    printf("  %.3f ms\n", stop_stopwatch(sw) * 1.0e3);
    printf("Image has dimension "BGREEN"%d x %d"NORMAL"\n", img_input->width, img_input->height);

    // handle magnification factor
    if (argc >= 3)
    {
        scalefactor = atoi(argv[2]);
    }
    printf("Using scaling factor of " BGREEN "%d" NORMAL "\n", scalefactor);

    // handle output file
    if (argc >= 4)
    {
        outname = argv[3];
    }

    printf(BCYAN "--------------------------------------------------------------------------------\n");
    printf("  Scale-up image:\n");
    printf("--------------------------------------------------------------------------------\n" NORMAL);

    printf(BCYAN "--------------------------------------------------------------------------------\n");
    printf("     using simple copy:\n");
    printf("--------------------------------------------------------------------------------\n" NORMAL);

    printf("Computing new image:\n");
    start_stopwatch(sw);
    img_output = NewBMPImage(scalefactor * img_input->width, scalefactor * img_input->height, img_input->bytes_per_pixel);
    copy_scale(img_input, img_output);
    printf("  %.3f ms\n", stop_stopwatch(sw) * 1.0e3);

    // saving output bitmap
    start_stopwatch(sw);
    // 2 for NIL-terminated, 4 for 'copy', 4 for '.bmp'
    length = strlen(outname) + 2 + 4 + 4;
    outfile = (char *)malloc(length * sizeof(char));
    sprintf(outfile, "%s_copy.bmp", outname);
    printf("Saving output to \"" BGREEN "%s" NORMAL "\"\n", outfile);
    WriteBMPImage(outfile, img_output);
    printf("  %.3f ms\n", stop_stopwatch(sw) * 1.0e3);
    DelBMPImage(img_output);
    free(outfile);

    printf(BCYAN "--------------------------------------------------------------------------------\n");
    printf("     using bicubic interpolation:\n");
    printf("--------------------------------------------------------------------------------\n" NORMAL);

    printf("Computing new image:\n");
    start_stopwatch(sw);
    img_output = NewBMPImage(scalefactor * img_input->width, scalefactor * img_input->height, img_input->bytes_per_pixel);
    // bicubic_interpolation(img_input, img_output);
    bicubic_hermite_interpolation(img_input, img_output);
    printf("  %.3f ms\n", stop_stopwatch(sw) * 1.0e3);

    // saving output bitmap
    start_stopwatch(sw);
    // 2 for NIL-terminated, 7 for 'bicubic', 4 for '.bmp'
    length = strlen(outname) + 2 + 7 + 4;
    outfile = (char *)malloc(length * sizeof(char));
    sprintf(outfile, "%s_bicubic.bmp", outname);
    printf("Saving output to \"" BGREEN "%s" NORMAL "\"\n", outfile);
    WriteBMPImage(outfile, img_output);
    printf("  %.3f ms\n", stop_stopwatch(sw) * 1.0e3);
    DelBMPImage(img_output);
    free(outfile);

    DelBMPImage(img_input);
    del_stopwatch(sw);

    // img_input = NewBMPImage(8, 4, 3);

    // for (int i = 0; i < 3 * img_input->width * img_input->height; i++)
    // {
    //     img_input->pixels[i] = 0;
    // }
    // int colld = img_input->width;
    // int i, j;

    // for (i = 0; i < img_input->height; i++)
    // {
    //     j = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 64;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 64;

    //     j = 1;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 127;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 127;

    //     j = 2;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 192;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 127;

    //     j = 3;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 255;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 127;

    //     j = 4;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 255;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 192;

    //     j = 5;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 192;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 192;

    //     j = 6;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 127;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 255;

    //     j = 7;
    //     img_input->pixels[3 * (j + i * colld) + BMP_R] = 64;
    //     img_input->pixels[3 * (j + i * colld) + BMP_G] = 0;
    //     img_input->pixels[3 * (j + i * colld) + BMP_B] = 255;
    // }

    // WriteBMPImage("min.bmp", img_input);

    return 0;
}
