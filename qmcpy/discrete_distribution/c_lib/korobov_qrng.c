/*
C function for computing a Korobov sequence.

References:
    
    [1] Marius Hofert and Christiane Lemieux (2019). 
    qrng: (Randomized) Quasi-Random Number Generators. 
    R package version 0.0-7.
    https://CRAN.R-project.org/package=qrng.

    [2] Faure, Henri, and Christiane Lemieux. 
    “Implementation of Irreducible Sobol’ Sequences in Prime Power Bases.” 
    Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.
*/

#include <stdlib.h>
#include "MRG63k3a.h"


/**
 * @title Generate n Points of a d-dimensional Korobov Sequence
 * @param n number of points
 * @param d dimension
 * @param generator vector of generator points
 * @param randomize string indicating whether the points are randomized
 * @param res pointer to the result matrix
 * @param seed seed for random number generator
 * @return void
 * @author Marius Hofert based on C. Lemieux's RandQMC
 */
EXPORT void korobov_qrng(int n, int d, int *generator, int randomize, double *res, long long seed)
{
	int i, j, ij;
	double U;
	double *aux;
	aux = (double *) calloc(d, sizeof(double));
	seed_MRG63k3a(seed);

	/* Init */
	for(j=0; j<d; j++){
		aux[j] = generator[j] / ((double) n);
		res[j*n] = 0.0; /* case i = 0 below */
	}

	/* Generate points */
	for(i=1; i<n; i++){ /* omit i=0 as done in init above */
		for(j=0; j<d; j++){
			ij = j*n+i;
			res[ij] = res[j*n + (i-1)] + aux[j];
			if(res[ij] > 1) res[ij] = res[ij] - 1.0;
		}
	}

	/* Randomization */
	if(randomize == 1) {
		for(j=0; j<d; j++){
			U = MRG63k3a(); /* 63 bit U(0,1) random number */ 
			for(i=0; i<n; i++){
				ij = j*n+i;
				res[ij] = res[ij] + U;
				if(res[ij] > 1) res[ij] = res[ij] - 1.0;
			}
		}
	}
	free(aux);
}
