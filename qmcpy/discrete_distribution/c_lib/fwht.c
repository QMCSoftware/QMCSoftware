/*
Fast Walsh Hadamard transform.
*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "export_ctypes.h"

EXPORT void fwht_copy(unsigned int n, double *src, double *dst)
{
    double *a = (double *)calloc(n, sizeof(double));
    double *b = (double *)calloc(n, sizeof(double));

    void *tmp;
    memcpy(a, src, sizeof(double) * n);

    // Fast Walsh Hadamard Transform.
    int i, s;
    unsigned int j;
    for (i = n >> 1; i > 0; i >>= 1)
    {
        for (j = 0; j < n; j++)
        {
            s = j / i % 2;
            b[j] = a[(s ? -i : 0) + j] + (s ? -1 : 1) * a[(s ? 0 : i) + j];
        }
        tmp = a;
        a = b;
        b = tmp;
    }

    memcpy(dst, a, sizeof(double) * n);
    free(a);
    free(b);
}

EXPORT void fwht_normalize(int n, int *src)
{
    int i;
    for (i = 0; i < n; i++)
        src[i] /= n;
}

// (a,b) -> (a+b,a-b) without overflow
void rotate(double *a, double *b)
{
    static double t;
    t = *a;
    *a = *a + *b;
    *b = t - *b;
}

// Integer log2
long ilog2(long x)
{
    long l2 = 0;
    for (; x; x >>= 1)
        ++l2;
    return l2;
}

/**
 * Fast Walsh-Hadamard transform: no copy used
 * Ref: https://github.com/sheljohn/WalshHadamard/blob/master/fwht.h
 */
EXPORT void fwht_inplace(unsigned long n, double *data)
{
    const long l2 = ilog2(n) - 1;

    for (long i = 0; i < l2; ++i)
    {
        for (long j = 0; j < (1 << l2); j += 1 << (i + 1))
            for (long k = 0; k < (1 << i); ++k)
                rotate(&data[j + k], &data[j + k + (1 << i)]);
    }
}