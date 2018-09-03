"""
dt_integrate.py
doctests for integrate

Examples

Example 1: integrate over given default instances of KeisterFun,
IIDDistribution, and CLTStopping.
>>> from integrate import integrate as integrate
>>> from KeisterFun import KeisterFun as KeisterFun
>>> from IIDDistribution import IIDDistribution as IID
>>> from CLTStopping import CLTStopping as CLTStopping
>>> from meanVarData import meanVarData as meanVar
>>> funObj = KeisterFun()
>>> distribObj = IIDDistribution()
>>> stopObj = CLTStopping()
>>> datObj = meanVarData()
>>> [solution, dataObj] = integrate(funObj, distribObj, stopObj, datObj)

solution =

  0.4310

dataObj =***

         muhat: 0.4310
        sighat: 0.2611
        nSigma: 1024
           nMu: 6516
      solution: 0.4310
         stage: 'done'
         prevN: 1024
         nextN: 7540
      timeUsed: 0.00***
  nSamplesUsed: 7540
    errorBound: [0.4210 0.4410]
         costF: ***e-04
"""
