"""
>>> from numpy import arange
>>> from measure import measure
>>> measure_1D = measure(measureName='stdGaussian')
>>> print(measure_1D)
measure with properties:
    measureName: stdGaussian
    dimension: [2]
    domainCoord: None
    domainShape: None
    measureData: None
    variance: None
    timeVector: None
    measure_list: [measure with properties:
    measureName: stdGaussian
    dimension: 2
    domainCoord: []
    domainShape:
    measureData: []
    variance: None
    timeVector: None
    measure_list: []]

>>> timeVecs = [arange(1/4,5/4,1/4),arange(1/16,17/16,1/16),arange(1/64,65/64,1/64)]
>>> measure_3D = measure(measureName='BrownianMotion',timeVector=timeVecs)
>>> print(measure_3D)
measure with properties:
    measureName: BrownianMotion
    dimension: [2]
    domainCoord: None
    domainShape: None
    measureData: None
    variance: None
    timeVector: [array([0.25, 0.5 , 0.75, 1.  ]), array([0.0625, 0.125 , 0.1875, 0.25  , 0.3125, 0.375 , 0.4375, 0.5   ,
       0.5625, 0.625 , 0.6875, 0.75  , 0.8125, 0.875 , 0.9375, 1.    ]), array([0.015625, 0.03125 , 0.046875, 0.0625  , 0.078125, 0.09375 ,
       0.109375, 0.125   , 0.140625, 0.15625 , 0.171875, 0.1875  ,
       0.203125, 0.21875 , 0.234375, 0.25    , 0.265625, 0.28125 ,
       0.296875, 0.3125  , 0.328125, 0.34375 , 0.359375, 0.375   ,
       0.390625, 0.40625 , 0.421875, 0.4375  , 0.453125, 0.46875 ,
       0.484375, 0.5     , 0.515625, 0.53125 , 0.546875, 0.5625  ,
       0.578125, 0.59375 , 0.609375, 0.625   , 0.640625, 0.65625 ,
       0.671875, 0.6875  , 0.703125, 0.71875 , 0.734375, 0.75    ,
       0.765625, 0.78125 , 0.796875, 0.8125  , 0.828125, 0.84375 ,
       0.859375, 0.875   , 0.890625, 0.90625 , 0.921875, 0.9375  ,
       0.953125, 0.96875 , 0.984375, 1.      ])]
    measure_list: [measure with properties:
    measureName: BrownianMotion
    dimension: 4
    domainCoord: []
    domainShape:
    measureData: []
    variance: None
    timeVector: [0.25 0.5  0.75 1.  ]
    measure_list: [], measure with properties:
    measureName: BrownianMotion
    dimension: 16
    domainCoord: []
    domainShape:
    measureData: []
    variance: None
    timeVector: [0.0625 0.125  0.1875 0.25   0.3125 0.375  0.4375 0.5    0.5625 0.625
 0.6875 0.75   0.8125 0.875  0.9375 1.    ]
    measure_list: [], measure with properties:
    measureName: BrownianMotion
    dimension: 64
    domainCoord: []
    domainShape:
    measureData: []
    variance: None
    timeVector: [0.015625 0.03125  0.046875 0.0625   0.078125 0.09375  0.109375 0.125
 0.140625 0.15625  0.171875 0.1875   0.203125 0.21875  0.234375 0.25
 0.265625 0.28125  0.296875 0.3125   0.328125 0.34375  0.359375 0.375
 0.390625 0.40625  0.421875 0.4375   0.453125 0.46875  0.484375 0.5
 0.515625 0.53125  0.546875 0.5625   0.578125 0.59375  0.609375 0.625
 0.640625 0.65625  0.671875 0.6875   0.703125 0.71875  0.734375 0.75
 0.765625 0.78125  0.796875 0.8125   0.828125 0.84375  0.859375 0.875
 0.890625 0.90625  0.921875 0.9375   0.953125 0.96875  0.984375 1.      ]
    measure_list: []]

"""