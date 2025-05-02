
# Discrete Distributions

The first $32$ points of each sequence are shown as purple starts, the next $32$ points are shown as green triangles, and the $128$ points after that are shown as blue circles. Notice the gaps and clusters of IID points compared to the more uniform coverage of LD sequences. We can use `QMCPy` to generate the LD digital net (in base $b=2$) as follows

```python 
>>> import qmcpy as qp 
>>> generator = qp.DigitalNetB2(dimension=2,seed=7)
>>> generator(8) # first 8 points in the sequence
array([[0.0715562 , 0.07784108],
       [0.81420169, 0.74485558],
       [0.31409299, 0.93233913],
       [0.57163057, 0.26535753],
       [0.15541309, 0.57499661],
       [0.89830224, 0.2439158 ],
       [0.39820498, 0.43143225],
       [0.6554989 , 0.76248017]])
>>> generator(8,16) # next 8 points in the sequence 
array([[0.03088897, 0.83362275],
       [0.77280156, 0.46942063],
       [0.272731  , 0.15687463],
       [0.53100149, 0.52116877],
       [0.1970359 , 0.33483624],
       [0.93870483, 0.97247967],
       [0.43861519, 0.66002569],
       [0.69712934, 0.02229023]])
```

The same API is available for the other LD sequences: `qp.Lattice`, `qp.DigitalNetB2`, and `qp.Halton`. A similar API for IID points is available in `qp.IIDStdUniform` (essentially a wrapper around [`numpy.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html))