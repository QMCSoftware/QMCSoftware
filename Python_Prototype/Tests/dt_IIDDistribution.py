"""
>>> from IIDDistribution import IIDDistribution as IIDDistribution; iid = IIDDistribution();
>>> iid.domainType
'box'
>>> iid.trueDistribution
'uniform'
>>> iid.dimension
2
>>> iid.domain
array([[0, 0],
       [1, 1]])

>>> iid = iid.initStreams(1,seed=10)
>>> iid.domainType
'box'
>>> iid.trueDistribution
'uniform'
>>> iid.dimension
2
>>> iid.domain
array([[0, 0],
       [1, 1]])
>>> iid.distribDataStream # doctest:+ELLIPSIS
[<randomstate.prng.mrg32k3a.mrg32k3a.RandomState object at ...

>>> x,w,a = iid.genDistrib(1, 2, 3, (1,2));
>>> w
1
>>> a
0.3333333333333333
>>> x
array([[0.18465602, 0.200767  ],
       [0.97176332, 0.84322271]])

"""
