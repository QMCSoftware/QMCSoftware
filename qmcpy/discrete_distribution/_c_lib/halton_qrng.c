/* C function for computing a generalized Halton sequence. 

References:
    
    [1] Marius Hofert and Christiane Lemieux (2019). 
    qrng: (Randomized) Quasi-Random Number Generators. 
    R package version 0.0-7.
    https://CRAN.R-project.org/package=qrng.

    [2] Faure, Henri, and Christiane Lemieux. 
    “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases.” 
    Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "export_ctypes.h"

#define ghaltonMaxDim 360

/* Primes for ghalton() */
static int primes[ghaltonMaxDim] =
    {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101,
     103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
     199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
     313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
     433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557,
     563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
     673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
     811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
     941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049,
     1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153,
     1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277,
     1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381,
     1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487,
     1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597,
     1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699,
     1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823,
     1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949,
     1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063,
     2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161,
     2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293,
     2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393,
     2399, 2411, 2417, 2423};

/* Scrambling factors for ghalton() */
static int permTN2[ghaltonMaxDim] =
    {1, 1, 3, 3, 4, 9, 7, 5, 9, 18, 18, 8, 13, 31, 9, 19, 36, 33, 21, 44, 43, 61, 60, 56, 26, 71, 32, 77, 26, 95,
     92, 47, 29, 61, 57, 69, 115, 63, 92, 31, 104, 126, 50, 80, 55, 152, 114, 80, 83, 97, 95, 150, 148, 55,
     80, 192, 71, 76, 82, 109, 105, 173, 58, 143, 56, 177, 203, 239, 196, 143, 278, 227, 87, 274, 264, 84,
     226, 163, 231, 177, 95, 116, 165, 131, 156, 105, 188, 142, 105, 125, 269, 292, 215, 182, 294, 152,
     148, 144, 382, 194, 346, 323, 220, 174, 133, 324, 215, 246, 159, 337, 254, 423, 484, 239, 440, 362,
     464, 376, 398, 174, 149, 418, 306, 282, 434, 196, 458, 313, 512, 450, 161, 315, 441, 549, 555, 431,
     295, 557, 172, 343, 472, 604, 297, 524, 251, 514, 385, 531, 663, 674, 255, 519, 324, 391, 394, 533,
     253, 717, 651, 399, 596, 676, 425, 261, 404, 691, 604, 274, 627, 777, 269, 217, 599, 447, 581, 640,
     666, 595, 669, 686, 305, 460, 599, 335, 258, 649, 771, 619, 666, 669, 707, 737, 854, 925, 818, 424,
     493, 463, 535, 782, 476, 451, 520, 886, 340, 793, 390, 381, 274, 500, 581, 345, 363, 1024, 514,
     773, 932, 556, 954, 793, 294, 863, 393, 827, 527, 1007, 622, 549, 613, 799, 408, 856, 601, 1072,
     938, 322, 1142, 873, 629, 1071, 1063, 1205, 596, 973, 984, 875, 918, 1133, 1223, 933, 1110, 1228,
     1017, 701, 480, 678, 1172, 689, 1138, 1022, 682, 613, 635, 984, 526, 1311, 459, 1348, 477, 716,
     1075, 682, 1245, 401, 774, 1026, 499, 1314, 743, 693, 1282, 1003, 1181, 1079, 765, 815, 1350,
     1144, 1449, 718, 805, 1203, 1173, 737, 562, 579, 701, 1104, 1105, 1379, 827, 1256, 759, 540,
     1284, 1188, 776, 853, 1140, 445, 1265, 802, 932, 632, 1504, 856, 1229, 1619, 774, 1229, 1300,
     1563, 1551, 1265, 905, 1333, 493, 913, 1397, 1250, 612, 1251, 1765, 1303, 595, 981, 671, 1403,
     820, 1404, 1661, 973, 1340, 1015, 1649, 855, 1834, 1621, 1704, 893, 1033, 721, 1737, 1507, 1851,
     1006, 994, 923, 872, 1860};

/**
 * @title Generate n Points of a d-dimensional Generalized Halton Sequence
 * @param n number of points
 * @param d dimension
 * @param n0 number of points to skip
 * @param method int indicating which sequence is generated
 *        (generalized Halton (1) or (plain) Halton (0))
 * @param res pointer to the result matrix
 * @param randu_d_32 seeds for random number generator
 * @param dvec dimensions to use
 * @return void
 * @author Marius Hofert based on C. Lemieux's RandQMC
 */
EXPORT void halton_qrng(int n, int d, int n0, int generalized, double *res, double *randu_d_32, int *dvec)
{
    static int perm[ghaltonMaxDim];
    int base, i, j, k, l, maxindex, f, start;
    double u;
    unsigned int tmp;
    unsigned int shcoeff[ghaltonMaxDim][32]; /* the coefficients of the shift */
    unsigned int coeff[32];

    /* Init */
    for (j = 0; j < d; j++)
    {

        base = primes[dvec[j]];
        u = 0;
        for (k = 31; k >= 0; k--)
        {
            shcoeff[j][k] = (int)(base * randu_d_32[j * 32 + k]);
            u += shcoeff[j][k];
            u /= base;
        }
        if (n0 == 0)
        {
            res[j * n] = u;
        }
    }

    /* Main */
    if (!generalized)
    {
        for (j = 0; j < d; j++)
        {
            perm[j] = 1;
        }
    }
    else
    {
        for (j = 0; j < d; j++)
        {
            perm[j] = permTN2[dvec[j]];
        }
    }
    if (n0 == 0)
    {
        start = 1;
    }
    else
    {
        start = n0;
    }
    for (i = start; i < (n + n0); i++)
    {
        for (j = 0; j < d; j++)
        {
            tmp = i;
            base = primes[dvec[j]];              /* (j+1)st prime number for this dimension */
            memset(&coeff, 0, sizeof(int) * 32); /* clear the coefficients */

            /* Find i in the prime base */
            k = 0;
            while ((tmp > 0) && (k < 32))
            {
                coeff[k] = tmp % base;
                tmp /= base;
                k++;
            }
            maxindex = k;
            for (l = maxindex + 1; l < 32; l++)
            {
                coeff[l] = 0;
            }
            u = 0.0;
            k = 31;
            f = perm[j];
            while (k >= 0)
            {
                u += (f * coeff[k] + shcoeff[j][k]) % base;
                u /= base;
                k--;
            }
            if (n0 == 0)
            {
                res[j * n + i] = u;
            }
            else
            {
                res[j * n + (i - n0)] = u;
            }
        }
    }
}

/*
int main()
{
    int n = 4, d = 3, n0 = 4, generalize = 1, skip = 0, seed = 7;
    double *res = (double *)calloc(d * n, sizeof(double));
    halton_qrng(n, d, n0, generalize, res, seed);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < d; j++)
        {
            printf("%.3f\t", res[j * n + i]);
        }
        printf("\n");
    }
    return (0);
}
*/