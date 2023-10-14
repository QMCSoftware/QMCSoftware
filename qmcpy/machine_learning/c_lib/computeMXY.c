#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "export_ctypes.h"


/* For t'=0,...,t, the function computes the number N_t' of all s-dimensional
 * multi-indices mvec' such that |mvec'|=t' and mvec'<= mvec component-wise.
 * The output is a t+1 dimensional vector N=(N_0,N_1,...,N_t).*/
void computeN(int s, int t, int* mvec, int** result, int** tmp);

/* Looks for the largest possibe number m in {lb,...,ub} such that
 * 2^m a- floor(2^m b) <1 and returns -1 if no such number exists*/
int searchmaxm(double a, double b, int ub, int base);

/* Computes the numbers S_t(z) */
void computeS(int s, int N, int  t, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2);

/* Computes the binomial coefficient (n k) and divides by N*/
double nchoosekbyN(int n, int k, double N);

/* double* computeWeights(int m, int mp, int s, int N, int Nqmc, double* px, double* pz, double* py); */

void print_array_values(double *arr, int size, int n,  char *name) {
    printf("The first few and last %d values of array %s are:\n", n, name);
    for (int i = 0; i < n && i < size; i++) {
            printf("%.3f ", arr[i]);
        }
    printf(" ... ");
    for (int i = size - n; i < size; i++) {
        printf("%.3f ", arr[i]);
    }
    printf("\n");
}


EXPORT double* computeWeights(int m, int mp, int s, int N, int Nqmc, double* px, double* pz, double* py){
  int base=2;
  /*
  printf("DEBUG m = %d, mp = %d, base = %d, s = %d, N = %d, Nqmc = %d \n", m, mp, base, s, N, Nqmc);
  print_array_values(px, N * s, 3, "px");
  print_array_values(pz, s * Nqmc, 3, "pz");
  print_array_values(py, N, 3, "py");
  */
  int outs = 1;

  int* result1 = malloc((m+1)*sizeof(int));
  double* result2 = malloc((m+1)*sizeof(double));
  int* tmp1 = malloc((m+1)*sizeof(int));
  int* tmp2 = malloc((m+1)*sizeof(int));
  int* mvec = malloc(s*sizeof(int));

    /* compute M_{m,mp}(f,x,y) */
    double M=0;
    int q,ell;
    int minsm=s-1; /*contains min(s-1,m) */
    if(m<minsm){minsm=m;}

    double* weights=(double*)malloc((1+outs)*Nqmc*sizeof(double));   /* return vector of weights */

   for(ell=0;ell<Nqmc;++ell){

       computeS(s, N, m, base, &pz[ell*s], px, py, result1, result2, mvec, tmp1, tmp2);
        double Mtmp1 =0;
        double Mtmp2 =0;
        for(q=0;q<=minsm;++q){
            double tmp= pow(-1,q)*nchoosekbyN(s-1,q,N*pow(base,mp-m+q));
            Mtmp1 += tmp*result1[m-q];
            Mtmp2 += tmp*result2[m-q];
            //printf("\nell = %d, q = %d, tmp = %.3f, result1[m-q] = %.3f, result2[m-q] = %.3f, Mtmp1 = %.3f, Mtmp2 = %.3f", ell, q, tmp, result1[m-q], result2[m-q], Mtmp1, Mtmp2);
        }
        weights[ell]=Mtmp1;
        weights[ell+Nqmc]=Mtmp2;
   }
   //printf("\n");
   //print_array_values(weights, (1+outs)*Nqmc, 5, "weights");
   return weights;
}


int searchmaxm(double a, double b, int ub, int base){
    int m=0;

    if(b>a){
        double tmp = a;
        a=b;
        b=tmp;
    }
    for(m=0;m <= ub;++m){
        if(a-(int)(b)>=1){
            return m-1;
        }
        a*=base;
        b*=base;
    }
    return ub;
}



void computeN(int s, int t, int* mvec, int** result,int** tmp){
    int i=0,j=0,k=0;
    int sum=0;
    int* p1,*p2,*p3;
    /* in each iteration of the main loop, the function reads from p1 and
     * writes in p2. After each iteration, p1 and p2 are swaped. */
    p1=*result;
    p2=*tmp;
    /* compute min(t,mvec[0]) */
    int min=mvec[0];
    if(t<min){ min=t;}
    /* initialize p1 */
    for(i=0;i<=min;++i){
        p1[i]=1;
    }
    for(i=min+1;i<t+1;++i){
        p1[i]=0;
    }
    /* Main loop */
    for(j=1;j<s;++j){
        if(mvec[j]<0){
            for(i=0;i<t+1;++i){
                p1[i]=0;
            }
            break;
        }
        sum=p1[0];
        p2[0]=sum;
        for(k=1;k<t+1;++k){
            sum+= p1[k];
            if(k-mvec[j]>0){
                sum-=p1[k-mvec[j]-1];
            }
            p2[k]=sum;
        }
        /* swapping read and write pointers */
        p3=p2;
        p2=p1;
        p1=p3;
    }
    /* returning the result. Don't forget to free tmp*/
    *result = p1;
    *tmp = p2;
}

void computeS(int s, int N, int t, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2){
    int i=0,j=0,k=0;
    for(i=0;i<t+1;++i){
        result1[i]=0;
        result2[i]=0;
    }
    for(i=0;i<N;++i){
        int tmp=0;
        int tmpsum=0;
        for(j=0;j<s;++j){
             tmp=searchmaxm(z[j],x[i + N*j],t,base);

             tmpsum+=tmp;
             mvec[j]=tmp;
             /*if(tmp==-1){
                 tmpsum=-1;
                 break;
             } */
        }
        if(1){
            for(j=0;j<t+1;++j){
                tmp1[j]=0;
                tmp2[j]=0;
            }
            computeN(s,t,mvec, &tmp1,&tmp2);
            for(j=0;j<t+1;++j){
                result1[j]+=tmp1[j];
                result2[j]+=y[i]*tmp1[j];
            }

        }
    }

}

double nchoosekbyN(int n, int k, double N){
    if(k>n/2){
        return nchoosekbyN(n,n-k,N);
    }
    double val=1.0/N;
    int i;
    for(i=1;i<=k;++i){
        val = val*(n-k+i)/i;
    }
    return val;    
}