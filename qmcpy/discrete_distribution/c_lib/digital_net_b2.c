#include "export_ctypes.h"
#include <stdlib.h>
#include <math.h>

EXPORT int get_unsigned_long_size() { return sizeof(unsigned long); }
EXPORT int get_unsigned_long_long_size() { return sizeof(unsigned long long); }

EXPORT int gen_digitalnetb2(
    unsigned long n0,           /* starting index in sequence. Must be a power of 2 if not using Graycode ordering */
    unsigned long n,            /* n: sample indicies include n0:(n0+n). Must be a power of 2 if not using Graycode ordering */
    unsigned int d,             /* dimension supported by generating vector */
    unsigned int graycode,      /* Graycode flag */
    unsigned int m_max,         /* 2^m_max is the maximum number of samples supported */
    unsigned int t2_max,        /* rows of znew, also the number of bits in each int of znew */
    unsigned long long *znew,   /* generating vector with shape d_max x m_max from set_digitalnetb2_randomizations */
    unsigned int set_rshift,    /* random shift flag */
    unsigned long long *rshift, /* length d vector of random digital shifts from set_digitalnetb2_randomizations */
    double *x,                  /* unrandomized points with shape n x d initialized to 0*/
    double *xr)                 /* randomized points with shape n x d initialized to 0*/
{
    /*
    Digital Net Generator by alegresor 

    References:
            
        [1] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
        DOI:https://doi.org/10.1145/42288.214372
        
    Error Codes:
        1) using natural ordering (graycode=0) and n0 and/or (n0+n) is not 0 or a power of 2
        2) n0+n exceeds 2^m_max
    */
    if ((n == 0) || (d == 0))
    {
        return (0);
    }
    if ((graycode == 0) && (((n0 != 0) && fmod(log(n0) / log(2), 1) != 0) || (fmod(log(n0 + n) / log(2), 1) != 0)))
    {
        /* for natural ordering, require n0 and (n0+n) be either 0 or powers of 2 */
        return (1);
    }
    if ((n0 + n) > ldexp(1, m_max))
    {
        /* too many samples */
        return (2);
    }
    double scale = ldexp(1, -1 * t2_max);
    unsigned int j, m, rm1bit;
    unsigned long long i, i_gc, idx;
    unsigned long long *xc = (unsigned long long *)calloc(d, sizeof(unsigned long long)); /* binary current point before scaling */
    /* n0 point */
    i = n0;
    i_gc = i ^ (i >> 1);
    if (graycode)
    {
        idx = i;
    }
    else
    {
        idx = i_gc;
    }
    for (m = 0; m < m_max; m++)
    {
        if (i_gc == 0)
        {
            break;
        }
        if (i_gc & 1)
        {
            for (j = 0; j < d; j++)
            {
                xc[j] = xc[j] ^ znew[j * m_max + m];
            }
        }
        i_gc >>= 1;
    }
    for (j = 0; j < d; j++)
    {
        x[(idx - n0) * d + j] = ((double)xc[j]) * scale;
        if (set_rshift)
        {
            xr[(idx - n0) * d + j] = ((double)(xc[j] ^ rshift[j])) * scale;
        }
    }
    /* n0+1,...,n0+n points */
    for (i = n0 + 1; i < (n0 + n); i++)
    {
        /* rightmost 1 bit of i */
        rm1bit = 0;
        for (m = 0; m < m_max; m++)
        {
            if ((i >> m) & 1)
            {
                break;
            }
            rm1bit += 1;
        }
        if (graycode)
        {
            idx = i;
        }
        else
        {
            idx = i ^ (i >> 1);
        }
        for (j = 0; j < d; j++)
        {
            xc[j] = xc[j] ^ znew[j * m_max + rm1bit];
        }
        for (j = 0; j < d; j++)
        {
            x[(idx - n0) * d + j] = ((double)xc[j]) * scale;
            if (set_rshift)
            {
                xr[(idx - n0) * d + j] = ((double)(xc[j] ^ rshift[j])) * scale;
            }
        }
    }
    free(xc);
    return (0);
}

/*
int main()
{
    return (0);
}*/