"""
>>> from meanVarData import meanVarData as meanVarData
>>> mvd = meanVarData()
>>> print(mvd.__dict__)
{'muhat': [], 'sighat': [], 'nSigma': [], 'nMu': [], 'solution': nan, 'stage': 'begin', 'prevN': [], 'nextN': [], 'timeUsed': [], 'nSamplesUsed': [], 'errorBound': [-inf, inf], 'costF': []}

>>> mvd.timeStart()  # doctest:+ELLIPSIS
>>> mvd._meanVarData__timeStart # doctest:+ELLIPSIS
15...
"""