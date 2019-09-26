"""
>>> from algorithms.distribution.IIDDistribution import IIDDistribution
>>> from algorithms.distribution import measure
>>> distribObj = IIDDistribution(trueD=measure().stdGaussian(dimension=[3]))
>>> print(distribObj)
Constructing discreteDistribution with properties:
    distribData: None
    state: []
    trueD: Constructing measure with properties:
                   domainShape:
                   domainCoord: []
                   measureData: {}
                   measureName: stdGaussian
                   dimension: [3]
                   measure_list:
                       measure_list[0] with properties:
                           domainShape:
                           domainCoord: []
                           measureData: {}
                           dimension: 3
                           measureName: stdGaussian
    distrib_list:
        distrib_list[0] with properties:
            distribData: None
            state: []
            trueD: Constructing measure with properties:
                        domainShape:
                        domainCoord: []
                        measureData: {}
                        dimension: 3
                        measureName: stdGaussian
                        measure_list:

"""
