"""
>>> from meanVar import meanVar as meanVar
>>> mvd = meanVar(); print(mvd.__dict__)
{'muhat': [], 'sighat': [], 'nSigma': [], 'nMu': [], 'solution': nan, 'stage': 'begin', 'prevN': [], 'nextN': [], 'timeUsed': [], 'nSamplesUsed': [], 'errorBound': [-inf, inf], 'costF': []}


>>> mvd.timeStart()  # doctest:+ELLIPSIS
>>> mvd._meanVar__timeStart  # doctest:+ELLIPSIS
1...

"""