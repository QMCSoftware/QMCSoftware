#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include <math.h>

/* For t'=0,...,t, the function computes the number N_t' of all s-dimensional 
 * multi-indices mvec' such that |mvec'|=t' and mvec'<= mvec component-wise.
 * The output is a t+1 dimensional vector N=(N_0,N_1,...,N_t).*/
void computeN(int s, int t, int* mvec, int** result, int** tmp); // This is present in computeMXY

/* Looks for the largest possibe number m in {lb,...,ub} such that
 * 2^m a- floor(2^m b) <1 and returns -1 if no such number exists*/
int searchmaxm(double a, double b, int ub, int base); //This is present in compyeMXY.c

/* Computes the numbers S_t(z) */
void computeS(int s, int N, int  t, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2);

/* Computes the binomial coefficient (n k) and divides by N*/
double nchoosekbyN(int n, int k, double N);

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]){
    
    
    
    int m = mxGetScalar(prhs[0]);  /* parameter m */
    int mp = mxGetScalar(prhs[1]); /* parameter m' */
    int base = mxGetScalar(prhs[2]);
    double* px = mxGetPr(prhs[3]); /* (N x s) array of x-data points */
    double* pz = mxGetPr(prhs[4]); /* (s x 2^m') array of QMC points !transposed! */
    double* py = mxGetPr(prhs[5]); /* (N x 1) array  y-data points */
    
   
    int s = mxGetN(prhs[3]);
    int N = mxGetM(prhs[3]);
    int Nqmc = mxGetN(prhs[4]);
    mxArray* ret=mxCreateDoubleMatrix(Nqmc,2,mxREAL); /* return vector of weights */
    double* pret=mxGetPr(ret);
    
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
    
   for(ell=0;ell<Nqmc;++ell){
        computeS(s, N, m, base, &pz[ell*s], px, py, result1, result2, mvec, tmp1, tmp2);
        double Mtmp1 =0;
        double Mtmp2 =0;
        for(q=0;q<=minsm;++q){
            double tmp= pow(-1,q)*nchoosekbyN(s-1,q,N*pow(base,mp-m+q));
            Mtmp1 += tmp*result1[m-q];
            Mtmp2 += tmp*result2[m-q];
        }
        pret[ell]=Mtmp1;
        pret[ell+Nqmc]=Mtmp2;
    }
    
    /* return the computed weights */
    plhs[0]=ret;
}

/* The python gateway function */
void computeWeights(int m, int mp, int s, int N, int Nqmc, int base, double *px, double *pz, double *py){
    
    
    
  //int m = mxGetScalar(prhs[0]);  /* parameter m */
  //int mp = mxGetScalar(prhs[1]); /* parameter m' */
  //int base = mxGetScalar(prhs[2]);
  //double* px = mxGetPr(prhs[3]); /* (N x s) array of x-data points */
  //double* pz = mxGetPr(prhs[4]); /* (s x 2^m') array of QMC points !transposed! */  
  //double* py = mxGetPr(prhs[5]); /* (N x 1) array  y-data points */    
    
   
  //int s = mxGetN(prhs[3]);
  //int N = mxGetM(prhs[3]);
  //int Nqmc = mxGetN(prhs[4]);
    mxArray* ret=mxCreateDoubleMatrix(Nqmc,2,mxREAL); /* return vector of weights */
    double* pret=mxGetPr(ret);   
    
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
    
   for(ell=0;ell<Nqmc;++ell){
        computeS(s, N, m, base, &pz[ell*s], px, py, result1, result2, mvec, tmp1, tmp2);
        double Mtmp1 =0;
        double Mtmp2 =0;
        for(q=0;q<=minsm;++q){
            double tmp= pow(-1,q)*nchoosekbyN(s-1,q,N*pow(base,mp-m+q));         
            Mtmp1 += tmp*result1[m-q];
            Mtmp2 += tmp*result2[m-q];
        }
        pret[ell]=Mtmp1;
        pret[ell+Nqmc]=Mtmp2;
    }
    
    /* return the computed weights */   
    plhs[0]=ret;
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
