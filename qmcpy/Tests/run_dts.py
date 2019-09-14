''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
''' Run all doctests. Does NOT show differences on failures'''
from doctest import testfile

dt_objs = ['AsianCallFun','CLTStopping','IIDDistribution','integrate','KeisterFun','meanVarData','measure']
[print('\n%s: %s\n'%(obj,testfile('./dt_%s.py'%(obj),report=False))) for obj in dt_objs]