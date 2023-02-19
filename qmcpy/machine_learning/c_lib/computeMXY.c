#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "export_ctypes.h"


/* For t'=0,...,t, the function computes the number N_t' of all s-dimensional 
 * multi-indices mvec' such that |mvec'|=t' and mvec'<= mvec component-wise.
 * The output is a t+1 dimensional vector N=(N_0,N_1,...,N_t). Requires temp memory of size t+1 via pointer tmp*/
void computeN(int s, int t, int* mvec, int** result, int** tmp);

/* different (faster?) implementation of computeN*/
void computeNN(int s, int t, int* mvec, int** result, int** tmp);

/* helper function*/
int dim(int j);

/* Same as computeN but without the <= mvec restriction, i.e., mvec = (t,...,t) */
void computeN2(int s, int t, int** result,int** tmp);

/* Looks for the largest possibe number m in {lb,...,ub} such that
 * 2^m a- floor(2^m b) <1 and returns -1 if no such number exists*/
int searchmaxm(double a, double b, int ub, int base);

/* Computes the numbers S_t(z) from Algorithm 2 (result1) and T_t(z) from Algorithm 6 (result2). Requires temp memory of size t+1 via tmp1 and tmp2  */
void computeS(int s, int N, int  t, int outs, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2);
void computeSLinear(int s, int N, int  t,     int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2);

/* Computes the binomial coefficient (n k) and divides by N*/
double nchoosekbyN(int n, int k, double N);

void computeSLinear(int s, int N, int t, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2);

/* Computes the weights W_X,Y and W_X. 
nu ... \nu in the paper
m ... \ell in the paper
s ... dimension of data
N ... number of datapoints
Nqmc ... number of qmc points
outs ... output dimension of the neural network (weights W_X,Y are vector valued if outs>1)
px ... pointer to datapoints array
pz ... pointer to qmc points array
py ... pointer to y values array

Output is a pointer to a vector which contains the weights W_X (Nqmc entries), and then the dimensions of W_X,Y (Nqmc x outs entries) 
in the same order as the qmc points.
*/


EXPORT double* computeLinearWeights(int nu, int m, int mp, int s, int N, int Nqmc, int outs, double* px, double* pz, double* py){

  int* result1 = malloc((m+1)*sizeof(int));
  double* result2 = malloc((m+1)*sizeof(double));
  int* tmp1 = malloc((m+1)*sizeof(int));
  int* tmp2 = malloc((m+1)*sizeof(int));
  int* mvec = malloc(s*sizeof(int));
  //int mp = malloc((mp+1)*sizeof(int));
    /* compute M_{m,mp}(f,x,y) */
    int base=2;
    double M=0;
    int q,ell;
    int minsm=s-1; /*contains min(s-1,m) */
    double* pret;
    if(m<minsm){minsm=m;}

   for(ell=0;ell<Nqmc;++ell){
        computeSLinear(s, N, m, base, &pz[ell*s], px, py, result1, result2, mvec, tmp1, tmp2);
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
   return pret;
}

void computeSLinear(int s, int N, int t, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2){
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


EXPORT double* computeWeights(int nu, int m, int s, int N, int Nqmc, int outs, double* px, double* pz, double* py){

    int base = 2;
    double* weights;
    
    int* result1;
    double* result2;
    int* tmp1;
    int* coeff;
    int* tmp2;
    int* mvec;   
    double* Mtmp2;
    int i=0,j=0,sum=0;

    coeff = malloc(s*sizeof(int));
    tmp1 = malloc(s*sizeof(int));
    computeN2(s,s-1, &tmp1 ,&coeff);
 
    for(i=0;i<=s-1;++i){	
	sum=0;
	for(j=0;j<=i-1;++j){
	   sum+=coeff[j]*tmp1[i-j];
	}
	coeff[i]=1-sum;
    }
    free(tmp1);
	
    #pragma omp parallel shared(weights) private(result1,result2,tmp1,tmp2,mvec,Mtmp2)
    {
    weights=(double*)malloc((1+outs)*Nqmc*sizeof(double));/* return vector of weights */  
    
    result1 = malloc((nu+1)*(nu+1)*sizeof(int));
    result2 = malloc(outs*(nu+1)*(nu+1)*sizeof(double));
    tmp1 = malloc((nu+1)*(nu+1)*sizeof(int));
    tmp2 = malloc((nu+1)*(nu+1)*sizeof(int));
    mvec = malloc(s*sizeof(int));   
    Mtmp2 = malloc(outs*sizeof(double));

    /* compute M_{m,mp}(f,x,y) */
    double M=0;
    int q,ell,k,r;
    int minsm=s-1; /*contains min(s-1,nu) */

    if(nu<minsm)
      {
	minsm=nu;
      }
     #pragma omp for 
    for(ell=0;ell<Nqmc;++ell){
      computeS(s, N, nu, outs, base, &pz[ell*s], px, py, result1, result2, mvec, tmp1, tmp2);
	
      double Mtmp1 =0;
      for(k=0;k<outs;++k){
	Mtmp2[k]=0;
      }
      
      for(q=0;q<=minsm;++q){
	for(r=0;r<=nu-q;++r){
	  /*double tmp= pow(-1,q)*nchoosekbyN(s-1,q,N*pow(base,m-nu+q));*/
	  double tmp = coeff[q]*pow(base,-m+r)/N;   
	  Mtmp1 += tmp*result1[nu-q+(nu+1)*r];
	  for(k=0;k<outs;++k){
	    Mtmp2[k] += tmp*result2[nu-q+(nu+1)*r+(nu+1)*(nu+1)*k];
	  }
	}
      }
      weights[ell]=Mtmp1;
      for(k=0;k<outs;++k){
	weights[ell+(1+k)*Nqmc]=Mtmp2[k];
      }
    }
    free(result1); free(result2); free(tmp1); free(tmp2); free(mvec); free(Mtmp2);
    }
    /* return the computed weights */   
    return weights;
}

int dim(int j){
  return (int)(0.5+sqrt((double)j));
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

void computeN2(int s, int t, int** result,int** tmp){
  int i=0,j=0,k=0,l=0;
  int sum=0;
  int* p1,*p2,*p3;
  /* in each iteration of the main loop, the function reads from p1 and
   * writes in p2. After each iteration, p1 and p2 are swaped. */
  p1=*result;
  p2=*tmp;
  
  /* initialize p1 */
  for(i=0;i<t+1;++i){
    p1[i]=1;
  }
  
  /* Main loop */
  for(j=1;j<s;++j){
    int pj=dim(j);        
    p2[0]=0;
    for(k=0;k<t+1;++k){
      sum=0;    
      for(l=0;l<=k/pj;++l){
	sum += p1[k-pj*l];				
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

void computeNN(int s, int t, int* mvec, int** result,int** tmp){
  int i=0,j=0,k1=0,k2=0,l=0;
  int sum=0,start=0;
  int* p1,*p2,*p3;
  /* in each iteration of the main loop, the function reads from p1 and
   * writes in p2. After each iteration, p1 and p2 are swaped. */
  p1=*result;
  p2=*tmp;
  /* compute min(t,mvec[0]) */
  int min=mvec[0];
  if(t<min){ min=t;}
  /* initialize p1 */
  for(i=0;i<(t+1)*(t+1);++i){
    p1[i]=0;
  }
  for(i=0;i<=min;++i){
    p1[i+(t+1)*i]=1;
  }
  /* Main loop */
  for(j=1;j<s;++j){
    int pj=dim(j);
    for(k1=0;k1<t+1;++k1){
      for(k2=0;k2<t+1;++k2){
	sum=0;
	int tmp = mvec[j];
	if(k1/pj<tmp)
	  {
	    tmp=k1/pj;
	  }
	if(k2<tmp)
	  {
	    tmp=k2;
	  }    
	for(l=0;l<=tmp;++l){
	  sum += p1[k1-pj*l+(t+1)*(k2-l)];
	}     
	p2[k1+(t+1)*k2]=sum;
      }	   
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



void computeN(int s, int t, int* mvec, int** result,int** tmp){
  int i=0,j=0,k=0,l=0;
  int sum=0,start=0;
  int* p1,*p2,*p3;
  /* in each iteration of the main loop, the function reads from p1 and
   * writes in p2. After each iteration, p1 and p2 are swaped. */
  p1=*result;
  p2=*tmp;
  /* compute min(t,mvec[0]) */
  int min=mvec[0];
  if(t<min){
    min=t;
  }
  /* initialize p1 */
  for(i=0;i<=min;++i){
    p1[i]=1;
  }
  for(i=min+1;i<t+1;++i){
    p1[i]=0;
  }
  /* Main loop */
  for(j=1;j<s;++j){
    int pj=dim(j);
    if(mvec[j]<0){
      for(i=0;i<t+1;++i){
	p1[i]=0;
      }
      break;
    }
    
    p2[0]=0;
    for(k=0;k<t+1;++k){
      sum=0;
      int tmp = k-pj*mvec[j];
      if(tmp>0){
	start=tmp; 
      }else{
	start=0;
      }
      for(l=start;l<=k;++l){
	if((k-l)%pj==0){
	  sum += p1[l];
	}
	
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

void computeS(int s, int N, int t, int outs, int base, double* z, double* x, double* y, int* result1, double* result2, int* mvec, int* tmp1, int* tmp2){
  int i=0,j=0,k=0,j1=0,j2=0;
  for(i=0;i<(t+1)*(t+1);++i){
    result1[i]=0;
    for(k=0;k<outs;++k){
      result2[i+(t+1)*(t+1)*k]=0;
    }
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
      for(j=0;j<(t+1)*(t+1);++j){
	tmp1[j]=0;
	tmp2[j]=0;
      }
      computeNN(s,t,mvec, &tmp1,&tmp2);    
      
      for(j1=0;j1<t+1;++j1){    
	for(j2=0;j2<t+1;++j2){               
	  result1[j1+(t+1)*j2]+=tmp1[j1+(t+1)*j2];
	  for(k=0;k<outs;++k){
	    result2[j1+(t+1)*j2+(t+1)*(t+1)*k]+=y[i*outs + k]*tmp1[j1+(t+1)*j2];
	  }
	}
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
