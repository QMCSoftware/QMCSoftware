>>> from IIDDistribution import IIDDistribution;iid = IIDDistribution();print(iid.__dict__)
{'domain': array([[0, 0],
        [1, 1]]), 'domainType': 'box', 'dimension': 2, 'trueDistribution': 'uniform'}

>>> iid2 = iid.initStreams(1);print(iid2.__dict__)
{'domain': array([[0, 0],
        [1, 1]]), 'domainType': 'box', 'dimension': 2, 'trueDistribution': 'uniform', 'distribDataStream': [...]}


>>> x,w,a = iid2.genDistrib(1, 2, 3, (1,2));print(iid2.__dict__)
{'domain': array([[0, 0],
    [1, 1]]), 'domainType': 'box', 'dimension': 2, 'trueDistribution': 'uniform', 'distribDataStream': [<mtrand.RandomState object at 0x000002737F55DAF8>]}
>>> print(x,w,a)
1 0.3333333333333333
