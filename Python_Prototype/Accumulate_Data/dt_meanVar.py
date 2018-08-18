>>> from meanVarData import meanVarData;mvd = meanVarData();print(mvd.__dict__)
{'muhat': [], 'sighat': [], 'nSigma': [], 'nMu': [], 'solution': nan, 'stage': 'begin', 'prevN': [], 'nextN': [], 'timeUsed': [], 'nSamplesUsed': [], 'errorBound': [-inf, inf], 'costF': []}

>>> mvd = meanVarData();mvd.__timeStart # doctest:+ELLIPSIS
Traceback (most recent call last):
    ...
AttributeError: 'meanVarData' object has no attribute '__timeStart'
>>> mvd.timeStart()  # doctest:+ELLIPSIS
>>> mvd._meanVarData__timeStart  # doctest:+ELLIPSIS
1...
