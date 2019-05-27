''' Run all doctests. Does NOT show differences on failures'''
from doctest import testfile

dt_objs = ['AsianCallFun','CLTStopping','IIDDistribution','integrate','KeisterFun','meanVarData']
[print('%s: %s'%(obj,testfile('Tests/dt_%s.py'%(obj),report=False))) for obj in dt_objs]