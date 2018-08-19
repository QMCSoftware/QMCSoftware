"""
dt_integrate.py
doctests for integrate

Examples

Example 1: integrate over given default instances of KeisterFun,
IIDDistribution, and CLTStopping.
>>> import addpath # doctest:+ELLIPSIS
['/Users/terrya/ProgramData/QMCSoftware/Python_Prototype/Tests', '/Users/terrya/ProgramData/QMCSoftware/Python_Prototype', '/Applications/PyCharm CE.app/Contents/helpers/pycharm', '/Users/terrya/anaconda2/envs/py36/lib/python36.zip', '/Users/terrya/anaconda2/envs/py36/lib/python3.6', '/Users/terrya/anaconda2/envs/py36/lib/python3.6/lib-dynload', '/Users/terrya/anaconda2/envs/py36/lib/python3.6/site-packages']
['/Users/terrya/ProgramData/QMCSoftware/Python_Prototype/Tests', '/Users/terrya/ProgramData/QMCSoftware/Python_Prototype', '/Applications/PyCharm CE.app/Contents/helpers/pycharm', '/Users/terrya/anaconda2/envs/py36/lib/python36.zip', '/Users/terrya/anaconda2/envs/py36/lib/python3.6', '/Users/terrya/anaconda2/envs/py36/lib/python3.6/lib-dynload', '/Users/terrya/anaconda2/envs/py36/lib/python3.6/site-packages']
>>> from integrate import integrate
>>> from Keister import Keister
>> from IID import IID
>>> from CLT import CLT
>>> funObj = Keister()
>>> distribObj = IID()
>>> stopObj = CLT()
>>> [solution, dataObj] = integrate(funObj, distribObj, stopObj)

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
