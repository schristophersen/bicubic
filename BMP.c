
#include "BMP.h"

bitmap_image *NewBMPImage(int32 width, int32 height, int32 bytes_per_pixel)
{
    bitmap_image *img;

    img = (bitmap_image *)malloc(sizeof(bitmap_image));

    size_t unpaddedRowSize = width * bytes_per_pixel;
    size_t totalSize = unpaddedRowSize * height;
    img->pixels = (byte *)malloc(totalSize);
    img->width = width;
    img->height = height;
    img->bytes_per_pixel = bytes_per_pixel;

    return img;
}

bitmap_image *ReadBMPImage(const char *fileName)
{
    FILE *imageFile = fopen(fileName, "rb");
    int32 dataOffset;
    size_t count;

    bitmap_image *img = (bitmap_image *)malloc(sizeof(bitmap_image));
    byte **pixels = &(img->pixels);
    int32 *width = &(img->width);
    int32 *height = &(img->height);
    int32 *bytesPerPixel = &(img->bytes_per_pixel);

    fseek(imageFile, DATA_OFFSET_OFFSET, SEEK_SET);
    count = fread(&dataOffset, 4, 1, imageFile);
    assert(count > 0);
    fseek(imageFile, WIDTH_OFFSET, SEEK_SET);
    count = fread(width, 4, 1, imageFile);
    assert(count > 0);
    fseek(imageFile, HEIGHT_OFFSET, SEEK_SET);
    count = fread(height, 4, 1, imageFile);
    assert(count > 0);
    int16 bitsPerPixel;
    fseek(imageFile, BITS_PER_PIXEL_OFFSET, SEEK_SET);
    count = fread(&bitsPerPixel, 2, 1, imageFile);
    assert(count > 0);
    *bytesPerPixel = ((int32)bitsPerPixel) / 8;

    size_t paddedRowSize = (int)(4 * ceil((float)(*width) / 4.0f)) * (*bytesPerPixel);
    size_t unpaddedRowSize = (*width) * (*bytesPerPixel);
    size_t totalSize = unpaddedRowSize * (*height);
    *pixels = (byte *)malloc(totalSize);
    size_t i = 0;
    byte *currentRowPointer = *pixels + ((*height - 1) * unpaddedRowSize);
    for (i = 0; i < *height; i++)
    {
        fseek(imageFile, dataOffset + (i * paddedRowSize), SEEK_SET);
        count = fread(currentRowPointer, 1, unpaddedRowSize, imageFile);
        assert(count > 0);
        currentRowPointer -= unpaddedRowSize;
    }

    fclose(imageFile);

    return img;
}

void WriteBMPImage(const char *fileName, bitmap_image *img)
{
    assert(img != NULL);
    byte *pixels = img->pixels;
    int32 width = img->width;
    int32 height = img->height;
    int32 bytesPerPixel = img->bytes_per_pixel;

    FILE *outputFile = fopen(fileName, "wb");
    size_t count;
    //*****HEADER************//
    const char *BM = "BM";
    count = fwrite(&BM[0], 1, 1, outputFile);
    assert(count > 0);
    count = fwrite(&BM[1], 1, 1, outputFile);
    assert(count > 0);
    int paddedRowSize = (int)(4 * ceil((float)width / 4.0f)) * bytesPerPixel;
    int32 fileSize = paddedRowSize * height + HEADER_SIZE + INFO_HEADER_SIZE;
    count = fwrite(&fileSize, 4, 1, outputFile);
    assert(count > 0);
    int32 reserved = 0x0000;
    count = fwrite(&reserved, 4, 1, outputFile);
    assert(count > 0);
    int32 dataOffset = HEADER_SIZE + INFO_HEADER_SIZE;
    count = fwrite(&dataOffset, 4, 1, outputFile);
    assert(count > 0);

    //*******INFO*HEADER******//
    int32 infoHeaderSize = INFO_HEADER_SIZE;
    count = fwrite(&infoHeaderSize, 4, 1, outputFile);
    assert(count > 0);
    count = fwrite(&width, 4, 1, outputFile);
    assert(count > 0);
    count = fwrite(&height, 4, 1, outputFile);
    assert(count > 0);
    int16 planes = 1; // always 1
    count = fwrite(&planes, 2, 1, outputFile);
    assert(count > 0);
    int16 bitsPerPixel = bytesPerPixel * 8;
    count = fwrite(&bitsPerPixel, 2, 1, outputFile);
    assert(count > 0);
    // write compression
    int32 compression = NO_COMPRESION;
    count = fwrite(&compression, 4, 1, outputFile);
    assert(count > 0);
    // write image size (in bytes)
    int32 imageSize = width * height * bytesPerPixel;
    count = fwrite(&imageSize, 4, 1, outputFile);
    assert(count > 0);
    int32 resolutionX = 11811; // 300 dpi
    int32 resolutionY = 11811; // 300 dpi
    count = fwrite(&resolutionX, 4, 1, outputFile);
    assert(count > 0);
    count = fwrite(&resolutionY, 4, 1, outputFile);
    assert(count > 0);
    int32 colorsUsed = MAX_NUMBER_OF_COLORS;
    count = fwrite(&colorsUsed, 4, 1, outputFile);
    assert(count > 0);
    int32 importantColors = ALL_COLORS_REQUIRED;
    count = fwrite(&importantColors, 4, 1, outputFile);
    assert(count > 0);
    size_t i = 0;
    size_t unpaddedRowSize = width * bytesPerPixel;
    for (i = 0; i < height; i++)
    {
        size_t pixelOffset = ((height - i) - 1) * unpaddedRowSize;
        count = fwrite(&pixels[pixelOffset], 1, paddedRowSize, outputFile);
        assert(count > 0);
    }
    fclose(outputFile);
}

void DelBMPImage(bitmap_image *img)
{
    assert(img != NULL);
    if (img->pixels != NULL)
    {
        free(img->pixels);
    }
    free(img);
}

static void fill_patch(bitmap_image *img, int32 ioff, int32 joff, byte red, byte green, byte blue)
{
    assert(img != NULL);

    byte *pixels = img->pixels;
    int32 width = img->width;
    int32 height = img->height;
    int32 i, j;
    int32 iend, jend;
    float scalei, scalej;

    iend = ioff + 25;
    jend = joff + 25;

    assert(iend <= width);
    assert(jend <= height);

    for (j = joff; j < jend; j++)
    {
        scalej = 1.0f - ((25.0 - (float)(j - joff)) - 13.0f) * ((25.0 - (float)(j - joff)) - 13.0f) / 156.25f;
        for (i = ioff; i < iend; i++)
        {
            scalei = 1.0f - ((25.0 - (float)(i - ioff)) - 13.0f) * ((25.0 - (float)(i - ioff)) - 13.0f) / 156.25f;
            pixels[3 * (i + j * width) + BMP_R] = red * scalei * scalej;
            pixels[3 * (i + j * width) + BMP_G] = green * scalei * scalej;
            pixels[3 * (i + j * width) + BMP_B] = blue * scalei * scalej;
        }
    }
    for (i = ioff; i < iend; i++)
    {
        pixels[3 * (i + joff * width) + BMP_R] = 255;
        pixels[3 * (i + joff * width) + BMP_G] = 255;
        pixels[3 * (i + joff * width) + BMP_B] = 255;
        pixels[3 * (i + (jend - 1) * width) + BMP_R] = 255;
        pixels[3 * (i + (jend - 1) * width) + BMP_G] = 255;
        pixels[3 * (i + (jend - 1) * width) + BMP_B] = 255;
    }
    for (j = joff; j < jend; j++)
    {
        pixels[3 * (ioff + j * width) + BMP_R] = 255;
        pixels[3 * (ioff + j * width) + BMP_G] = 255;
        pixels[3 * (ioff + j * width) + BMP_B] = 255;
        pixels[3 * ((iend - 1) + j * width) + BMP_R] = 255;
        pixels[3 * ((iend - 1) + j * width) + BMP_G] = 255;
        pixels[3 * ((iend - 1) + j * width) + BMP_B] = 255;
    }
}

bitmap_image *make_test_example()
{
    bitmap_image *img;
    int32 width = 100;
    int32 height = 100;

    img = NewBMPImage(width, height, 3);

    fill_patch(img, 0, 0, 255, 0, 0);
    fill_patch(img, 25, 0, 200, 200, 30);
    fill_patch(img, 50, 0, 0, 0, 255);
    fill_patch(img, 75, 0, 255, 0, 255);

    fill_patch(img, 0, 25, 0, 50, 150);
    fill_patch(img, 25, 25, 50, 150, 255);
    fill_patch(img, 50, 25, 80, 255, 50);
    fill_patch(img, 75, 25, 200, 0, 50);

    fill_patch(img, 0, 50, 255, 255, 0);
    fill_patch(img, 25, 50, 0, 127, 0);
    fill_patch(img, 50, 50, 255, 0, 0);
    fill_patch(img, 75, 50, 0, 0, 255);

    fill_patch(img, 0, 75, 255, 127, 255);
    fill_patch(img, 25, 75, 0, 127, 127);
    fill_patch(img, 50, 75, 127, 255, 127);
    fill_patch(img, 75, 75, 255, 255, 127);

    return img;
}

void copy_scale(bitmap_image *input, bitmap_image *output)
{
    assert(input != NULL);
    assert(output != NULL);
    assert(input->bytes_per_pixel == output->bytes_per_pixel);

    int32 width = input->width; // width of input image
    int32 outwidth;
    int32 height = input->height; // height of input image
    int32 outheight;
    byte *inpixel = input->pixels;   // pixel array of input image
    byte *outpixel = output->pixels; // pixel array of output image
    int32 scale;
    int32 i, j, ii, jj, k, l;

    scale = output->width / width;
    assert(scale == output->height / height);
    assert(scale > 0);
    outwidth = width * scale;
    outheight = height * scale;

    for (j = 0; j < height; j++)
    {
        for (k = 0; k < scale; k++)
        {
            jj = j * scale + k;
            for (i = 0; i < width; i++)
            {
                for (l = 0; l < scale; l++)
                {
                    ii = i * scale + l;

                    outpixel[3 * (ii + jj * outwidth) + BMP_R] = inpixel[3 * (i + j * width) + BMP_R];
                    outpixel[3 * (ii + jj * outwidth) + BMP_G] = inpixel[3 * (i + j * width) + BMP_G];
                    outpixel[3 * (ii + jj * outwidth) + BMP_B] = inpixel[3 * (i + j * width) + BMP_B];
                }
            }
        }
    }
}

/* ------------------------------------------------------------
 * QR decomposition and solve functions
 * ------------------------------------------------------------ */

void axpy(const int n, const float alpha, const float *x, const int incx, float *y, const int incy)
{
    int i;

    for (i = 0; i < n; i++)
    {
        y[i * incy] = y[i * incy] + alpha * x[i * incx];
    }
}

float dot(const int n, const float *x, const int incx, const float *y, const int incy)
{
    int i;

    float res = 0.0f;

    for (i = 0; i < n; i++)
    {
        res += x[i * incx] * y[i * incy];
    }

    return res;
}

void scal(const int n, const float alpha, float *x, const int incx)
{
    int i;

    for (i = 0; i < n; i++)
    {
        x[i * incx] = alpha * x[i * incx];
    }
}

float nrm2(const int n, const float *x, const int incx)
{
    int i;

    float res = 0.0f;

    for (i = 0; i < n; i++)
    {
        res += x[i * incx] * x[i * incx];
    }

    return sqrtf(res);
}

void rsolve(const int n, const float *r, const int ldr, float *x)
{
    int k;

    for (k = n; k-- > 0;)
    {
        x[k] /= r[k + k * ldr];
        axpy(k, -x[k], r + k * ldr, 1, x, 1);
    }
}

void qtrans_eval(const int n, const float *a, const int lda, const float *tau, float *x)
{
    int k;
    float gamma;

    // Apply reflections in reverse order
    for (k = 0; k < n; k++)
    {
        // check if tau_k != 0.0
        if (tau[k] != 0.0)
        {
            // compute dot product of v_k with x
            gamma = x[k];
            gamma += dot(n - k - 1, a + k + 1 + k * lda, 1, x + k + 1, 1);
            // scale with tau_k
            gamma *= tau[k];

            // update x with gamma * v_k;
            x[k] -= gamma;
            axpy(n - k - 1, -gamma, a + k + 1 + k * lda, 1, x + k + 1, 1);
        }
    }
}

void qrsolve(const int n, const float *a, const int lda, const float *tau, float *b)
{
    qtrans_eval(n, a, n, tau, b);
    rsolve(n, a, n, b);
}

void decompqr(const int n, float *a, const int lda, float *tau)
{
    int j, k, cols;
    float norm_a, norm2_a, alpha, gamma;

    cols = n;

    for (k = 0; k < n; k++)
    {
        // ||a||_2
        norm_a = nrm2(n - k, a + k + k * lda, 1);
        // ||a||_2^2
        norm2_a = norm_a * norm_a;

        if (norm2_a > 1.0e-30f)
        {
            // compute alpha
            if (a[k + k * lda] < 0.0)
            {
                alpha = norm_a;
            }
            else
            {
                alpha = -norm_a;
            }
            // set tau
            tau[k] = fabsf(a[k + k * lda] - alpha);
            tau[k] *= tau[k];
            tau[k] /= (norm2_a - alpha * a[k + k * lda]);

            // set v, v_1 = 1.0
            norm_a = 1.0 / (a[k + k * lda] - alpha);
            scal(n - k - 1, norm_a, a + k + 1 + k * lda, 1);
            // set diagonal element of r
            a[k + k * lda] = alpha;

            // Update trailing matrix with Householder
            for (j = k + 1; j < cols; j++)
            {
                // compute dot product of v_k with jth column of trailing matrix
                gamma = a[k + j * lda];
                gamma += dot(n - k - 1, a + k + 1 + k * lda, 1, a + k + 1 + j * lda, 1);
                // scale with tau_k
                gamma *= tau[k];

                // update jth column of trailing matrix
                // with gamma * v_k;
                a[k + j * lda] -= gamma;
                axpy(n - k - 1, -gamma, a + k + 1 + k * lda, 1, a + k + 1 + j * lda, 1);
            }
        }
        else
        {
            tau[k] = 0.0;
        }
    }
}
int m = 3;
float xi[] = {0.125f, 0.375f, 0.625f, 0.875f};

static void setup_cubic_inter(float *x)
{
    float *V, *tau;
    int i, k;

    // Setup Vandermonde matrix
    V = (float *)malloc((m + 1) * (m + 1) * sizeof(float));

    // set first column to 1.0
    for (k = 0; k <= m; k++)
    {
        V[k] = 1.0;
    }
    // Compute remaining power of xi[k]
    for (i = 1; i <= m; i++)
    {
        for (k = 0; k <= m; k++)
        {
            V[k + i * (m + 1)] = V[k + (i - 1) * (m + 1)] * xi[k];
        }
    }

    // Factorize Vandermonde matrix
    tau = (float *)malloc((m + 1) * sizeof(float));
    decompqr(m + 1, V, m + 1, tau);

    // Solve system for interpolation coefficient in monomial basis
    qrsolve(m + 1, V, m + 1, tau, x);

    free(V);
    free(tau);
}

float *V = NULL;
float *tau = NULL;

static void setup_bicubic_inter(float *x)
{
    int i, j, k, l;
    int row, col, col1;
    int n = (m + 1) * (m + 1);

    if (V == NULL) // Setup Vandermonde matrix, if neccessary
    {
        assert(tau == NULL);

        // Setup Vandermonde matrix
        V = (float *)malloc(n * n * sizeof(float));

        // set first column to 1.0
        for (k = 0; k <= n; k++)
        {
            V[k] = 1.0;
        }

        // Compute remaining power of xi[k]
        for (i = 1; i <= m; i++)
        {
            for (l = 0; l <= m; l++)
            {
                for (k = 0; k <= m; k++)
                {
                    row = (k + l * (m + 1));
                    V[row + i * n] = V[row + (i - 1) * n] * xi[k];
                }
            }
        }

        // Compute remaining power of xi[l]
        for (j = 1; j <= m; j++)
        {
            for (i = 0; i <= m; i++)
            {
                col = (i + j * (m + 1));
                col1 = (i + (j - 1) * (m + 1));
                for (l = 0; l <= m; l++)
                {
                    for (k = 0; k <= m; k++)
                    {
                        row = (k + l * (m + 1));
                        V[row + col * n] = V[row + col1 * n] * xi[l];
                    }
                }
            }
        }

        // Factorize Vandermonde matrix
        tau = (float *)malloc(n * sizeof(float));
        decompqr(n, V, n, tau);
    }

    // Solve system for interpolation coefficient in monomial basis
    qrsolve(n, V, n, tau, x);
}

static inline float eval_cubic(const float *x, const float t)
{
    int i;

    float res = 0.0f;

    res = x[3];
    for (i = 3; i-- > 0;)
    {
        res = x[i] + t * res;
    }

    return res;
}

static inline void clamp_value(float *v, float lower, float upper)
{
    float val = *v;

    val = (val < lower) ? lower : val;
    val = (val > upper) ? upper : val;

    *v = val;
}

static inline void guard_load_float(byte *pix, int32 width, int32 height, float *val, int32 row, int32 col, byte color)
{
    row = (row < 0) ? 0 : row;
    row = (row >= width) ? width - 1 : row;
    col = (col < 0) ? 0 : col;
    col = (col >= height) ? height - 1 : col;

    *val = pix[3 * (row + col * width) + color];
}

void load_4x4_from_ij_value(bitmap_image *img, int32 i, int32 j, int32 color, float *val)
{
    assert(img != NULL);

    byte *pixels = img->pixels;
    int32 width = img->width;
    int32 height = img->height;

    val[0] = val[1] = val[2] = val[4] = 0.0f;
    val[4] = val[5] = val[6] = val[7] = 0.0f;
    val[8] = val[9] = val[10] = val[11] = 0.0f;
    val[12] = val[13] = val[14] = val[15] = 0.0f;

    // 1st row
    guard_load_float(pixels, width, height, &val[0], i + 0, j + 0, color);
    guard_load_float(pixels, width, height, &val[1], i + 1, j + 0, color);
    guard_load_float(pixels, width, height, &val[2], i + 2, j + 0, color);
    guard_load_float(pixels, width, height, &val[3], i + 3, j + 0, color);

    // 2nd row
    guard_load_float(pixels, width, height, &val[4], i + 0, j + 1, color);
    guard_load_float(pixels, width, height, &val[5], i + 1, j + 1, color);
    guard_load_float(pixels, width, height, &val[6], i + 2, j + 1, color);
    guard_load_float(pixels, width, height, &val[7], i + 3, j + 1, color);

    // 3rd row
    guard_load_float(pixels, width, height, &val[8], i + 0, j + 2, color);
    guard_load_float(pixels, width, height, &val[9], i + 1, j + 2, color);
    guard_load_float(pixels, width, height, &val[10], i + 2, j + 2, color);
    guard_load_float(pixels, width, height, &val[11], i + 3, j + 2, color);

    // 4th row
    guard_load_float(pixels, width, height, &val[12], i + 0, j + 3, color);
    guard_load_float(pixels, width, height, &val[13], i + 1, j + 3, color);
    guard_load_float(pixels, width, height, &val[14], i + 2, j + 3, color);
    guard_load_float(pixels, width, height, &val[15], i + 3, j + 3, color);
}

void load_4x4_hermite_value(bitmap_image *img, int32 i, int32 j, int32 color, float *val)
{
    assert(img != NULL);

    byte *pixels = img->pixels;
    int32 width = img->width;
    int32 height = img->height;

    val[0] = val[1] = val[2] = val[4] = 0.0f;
    val[4] = val[5] = val[6] = val[7] = 0.0f;
    val[8] = val[9] = val[10] = val[11] = 0.0f;
    val[12] = val[13] = val[14] = val[15] = 0.0f;

    // 1st col
    guard_load_float(pixels, width, height, &val[0], i - 1, j - 1, color);
    guard_load_float(pixels, width, height, &val[1], i - 1, j + 0, color);
    guard_load_float(pixels, width, height, &val[2], i - 1, j + 1, color);
    guard_load_float(pixels, width, height, &val[3], i - 1, j + 2, color);

    // 2nd col
    guard_load_float(pixels, width, height, &val[4], i + 0, j - 1, color);
    guard_load_float(pixels, width, height, &val[5], i + 0, j + 0, color);
    guard_load_float(pixels, width, height, &val[6], i + 0, j + 1, color);
    guard_load_float(pixels, width, height, &val[7], i + 0, j + 2, color);

    // 3rd col
    guard_load_float(pixels, width, height, &val[8], i + 1, j - 1, color);
    guard_load_float(pixels, width, height, &val[9], i + 1, j + 0, color);
    guard_load_float(pixels, width, height, &val[10], i + 1, j + 1, color);
    guard_load_float(pixels, width, height, &val[11], i + 1, j + 2, color);

    // 4th col
    guard_load_float(pixels, width, height, &val[12], i + 2, j - 1, color);
    guard_load_float(pixels, width, height, &val[13], i + 2, j + 0, color);
    guard_load_float(pixels, width, height, &val[14], i + 2, j + 1, color);
    guard_load_float(pixels, width, height, &val[15], i + 2, j + 2, color);
}

void bicubic_interpolation(bitmap_image *input, bitmap_image *output)
{
    assert(input != NULL);
    assert(output != NULL);
    assert(input->bytes_per_pixel == output->bytes_per_pixel);

    int32 width = input->width; // width of input image
    int32 outwidth;
    int32 height = input->height; // height of input image
    int32 outheight;
    byte *inpixel = input->pixels;   // pixel array of input image
    byte *outpixel = output->pixels; // pixel array of output image
    int32 scale;
    int32 subpix;
    float h, h2;
    int32 i, j, ii, jj, k, l;
    float u, v;
    float redval[16], greenval[16], blueval[16];
    float val[4];
    float newval[3];

    scale = output->width / width;
    assert(scale == output->height / height);
    assert(scale > 0);
    outwidth = width * scale;
    outheight = height * scale;
    subpix = 4 * scale;
    h = 1.0f / subpix;
    h2 = 0.5f * h;

    for (j = 0; j + 3 < height; j += 4)
    {
        jj = j * scale;
        for (i = 0; i + 3 < width; i += 4)
        {
            ii = i * scale;

            load_4x4_from_ij_value(input, i, j, BMP_R, redval); // read red values
            setup_bicubic_inter(redval);
            load_4x4_from_ij_value(input, i, j, BMP_G, greenval); // read green values
            setup_bicubic_inter(greenval);
            load_4x4_from_ij_value(input, i, j, BMP_B, blueval); // read blue values
            setup_bicubic_inter(blueval);

            for (k = 0; k < subpix; k++) // belongs to j
            {
                v = h2 + h * k;
                for (l = 0; l < subpix; l++) // belongs to i
                {
                    u = h2 + h * l;

                    // evaluate red color
                    val[0] = eval_cubic(redval + 0, u);
                    val[1] = eval_cubic(redval + 4, u);
                    val[2] = eval_cubic(redval + 8, u);
                    val[3] = eval_cubic(redval + 12, u);
                    newval[0] = eval_cubic(val, v);
                    clamp_value(&newval[0], 0.0f, 255.0f);

                    // evaluate green color
                    val[0] = eval_cubic(greenval + 0, u);
                    val[1] = eval_cubic(greenval + 4, u);
                    val[2] = eval_cubic(greenval + 8, u);
                    val[3] = eval_cubic(greenval + 12, u);
                    newval[1] = eval_cubic(val, v);
                    clamp_value(&newval[1], 0.0f, 255.0f);

                    // evaluate blue color
                    val[0] = eval_cubic(blueval + 0, u);
                    val[1] = eval_cubic(blueval + 4, u);
                    val[2] = eval_cubic(blueval + 8, u);
                    val[3] = eval_cubic(blueval + 12, u);
                    newval[2] = eval_cubic(val, v);
                    clamp_value(&newval[2], 0.0f, 255.0f);

                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_R] = (byte)newval[0];
                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_G] = (byte)newval[1];
                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_B] = (byte)newval[2];
                }
            }
        }
    }

    if (V != NULL)
    {
        free(V);
    }

    if (tau != NULL)
    {
        free(tau);
    }
}

// Product of A * B = C
// dim(a) = n x m
// dim(b) = m x p
// dim(c) = n x p
void gemm_nn(const float alpha,
             const int n, const int m, const int p,
             const float *a, const int lda,
             const float *b, const int ldb,
             float *c, const int ldc)
{
    int i, j, k;

    for (j = 0; j < p; j++)
    {
        for (k = 0; k < m; k++)
        {
            for (i = 0; i < n; i++)
            {
                c[i + j * ldc] += alpha * a[i + k * lda] * b[k + j * ldb];
            }
        }
    }
}

// Product of A * B = C
// dim(a) = n x m
// dim(b) = p x m
// dim(c) = n x p
void gemm_nt(const float alpha,
             const int n, const int m, const int p,
             const float *a, const int lda,
             const float *b, const int ldb,
             float *c, const int ldc)
{
    int i, j, k;

    for (k = 0; k < m; k++)
    {
        for (j = 0; j < p; j++)
        {
            for (i = 0; i < n; i++)
            {
                c[i + j * ldc] += alpha * a[i + k * lda] * b[j + k * ldb];
            }
        }
    }
}

// Inverse of T, colum major order
float Tinv[] = {1.0f, 0.0f, -3.0f, 2.0f,
                0.0f, 0.0f, 3.0f, -2.0,
                0.0f, 1.0f, -2.0f, 1.0f,
                0.0f, 0.0f, -1.0f, 1.0f};

/**
 * @brief Setup the coefficients for bicubic hermite interpolation
 *
 * @param x initially the 4x4 point grid, where only the inner 2x2 points
 *          build up the patch.
 *          The remaining points are necessary for computing the derivative
 */
void setup_bicubic_hermite_inter(float *x)
{
    float tmp[16], tmp2[16];
    int i, j;

    // copy patch values:
    tmp2[0 + 0 * 4] = x[1 + 1 * 4]; // P_00, row 0, col 0
    tmp2[1 + 0 * 4] = x[2 + 1 * 4]; // P_10, row 1, col 0
    tmp2[0 + 1 * 4] = x[1 + 2 * 4]; // P_01, row 0, col 1
    tmp2[1 + 1 * 4] = x[2 + 2 * 4]; // P_11, row 1, col 1

    // x-derivative values:
    tmp2[0 + 2 * 4] = 0.5f * (x[1 + 2 * 4] - x[1 + 0 * 4]); // Px_00
    tmp2[1 + 2 * 4] = 0.5f * (x[2 + 2 * 4] - x[2 + 0 * 4]); // Px_10
    tmp2[0 + 3 * 4] = 0.5f * (x[1 + 3 * 4] - x[1 + 1 * 4]); // Px_01
    tmp2[1 + 3 * 4] = 0.5f * (x[2 + 3 * 4] - x[2 + 1 * 4]); // Px_11

    // y-derivative values:
    tmp2[2 + 0 * 4] = 0.5f * (x[2 + 1 * 4] - x[0 + 1 * 4]); // Px_00
    tmp2[3 + 0 * 4] = 0.5f * (x[3 + 1 * 4] - x[1 + 1 * 4]); // Px_10
    tmp2[2 + 1 * 4] = 0.5f * (x[2 + 2 * 4] - x[0 + 2 * 4]); // Px_01
    tmp2[3 + 1 * 4] = 0.5f * (x[3 + 2 * 4] - x[1 + 2 * 4]); // Px_11

    // xy-derivative values:
    tmp2[2 + 2 * 4] = 0.25f * (x[2 + 2 * 4] - x[2 + 0 * 4] - x[0 + 2 * 4] + x[0 + 0 * 4]); // Px_00 at 11
    tmp2[3 + 2 * 4] = 0.25f * (x[3 + 2 * 4] - x[3 + 0 * 4] - x[1 + 2 * 4] + x[1 + 0 * 4]); // Px_10 at 21
    tmp2[2 + 3 * 4] = 0.25f * (x[2 + 3 * 4] - x[0 + 3 * 4] - x[2 + 1 * 4] + x[0 + 1 * 4]); // Px_01 at 12
    tmp2[3 + 3 * 4] = 0.25f * (x[3 + 3 * 4] - x[3 + 1 * 4] - x[1 + 3 * 4] + x[1 + 1 * 4]); // Px_11 at 22

    // clear tmp;
    for (i = 0; i < 16; i++)
    {
        tmp[i] = 0.0f;
    }

    gemm_nn(1.0, 4, 4, 4, Tinv, 4, tmp2, 4, tmp, 4);

    // clear tmp;
    for (i = 0; i < 16; i++)
    {
        x[i] = 0.0f;
    }

    gemm_nt(1.0, 4, 4, 4, tmp, 4, Tinv, 4, x, 4);
}

void bicubic_hermite_interpolation(bitmap_image *input, bitmap_image *output)
{
    assert(input != NULL);
    assert(output != NULL);
    assert(input->bytes_per_pixel == output->bytes_per_pixel);

    int32 width = input->width; // width of input image
    int32 outwidth;
    int32 height = input->height; // height of input image
    int32 outheight;
    byte *inpixel = input->pixels;   // pixel array of input image
    byte *outpixel = output->pixels; // pixel array of output image
    int32 scale;
    int32 subpix;
    float h, h2;
    int32 i, j, ii, jj, k, l;
    float u, v;
    float redval[16], greenval[16], blueval[16];
    float val[4];
    float newval[3];

    scale = output->width / width;
    assert(scale == output->height / height);
    assert(scale > 0);
    outwidth = width * scale;
    outheight = height * scale;
    subpix = 2 * scale;
    h = 1.0f / (scale);
    h2 = 0.5f * h;

    for (j = 0; j < height; j += 1)
    {
        jj = j * scale;
        for (i = 0; i < width; i += 1)
        {
            ii = i * scale;

            load_4x4_hermite_value(input, i, j, BMP_R, redval); // read red values
            setup_bicubic_hermite_inter(redval);
            load_4x4_hermite_value(input, i, j, BMP_G, greenval); // read green values
            setup_bicubic_hermite_inter(greenval);
            load_4x4_hermite_value(input, i, j, BMP_B, blueval); // read blue values
            setup_bicubic_hermite_inter(blueval);

            for (k = 0; k < scale; k++) // belongs to j
            {
                v = h * k;
                for (l = 0; l < scale; l++) // belongs to i
                {
                    u = h * l;

                    // evaluate red color
                    val[0] = eval_cubic(redval + 0, v);
                    val[1] = eval_cubic(redval + 4, v);
                    val[2] = eval_cubic(redval + 8, v);
                    val[3] = eval_cubic(redval + 12, v);
                    newval[0] = eval_cubic(val, u);
                    clamp_value(&newval[0], 0.0f, 255.0f);

                    // evaluate green color
                    val[0] = eval_cubic(greenval + 0, v);
                    val[1] = eval_cubic(greenval + 4, v);
                    val[2] = eval_cubic(greenval + 8, v);
                    val[3] = eval_cubic(greenval + 12, v);
                    newval[1] = eval_cubic(val, u);
                    clamp_value(&newval[1], 0.0f, 255.0f);

                    // evaluate blue color
                    val[0] = eval_cubic(blueval + 0, v);
                    val[1] = eval_cubic(blueval + 4, v);
                    val[2] = eval_cubic(blueval + 8, v);
                    val[3] = eval_cubic(blueval + 12, v);
                    newval[2] = eval_cubic(val, u);
                    clamp_value(&newval[2], 0.0f, 255.0f);

                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_R] = (byte)newval[0];
                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_G] = (byte)newval[1];
                    outpixel[3 * ((ii + l) + (jj + k) * outwidth) + BMP_B] = (byte)newval[2];
                }
            }
        }
    }

    if (V != NULL)
    {
        free(V);
    }

    if (tau != NULL)
    {
        free(tau);
    }
}