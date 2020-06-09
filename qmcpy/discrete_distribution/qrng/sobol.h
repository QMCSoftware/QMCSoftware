/* C header for sobol.c *******************************************************/

#ifndef sobol_H
#define sobol_H

#define sobolMaxDim 16510
#define sobolMaxDegree 17
#define sobolMaxCol 32

void sobol(int n, int d, int randomize, double *res, int skip, int graycode, long seed);

#endif
