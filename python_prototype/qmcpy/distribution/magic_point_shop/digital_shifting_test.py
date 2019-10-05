#!/usr/bin/env python_prototype
from __future__ import print_function
from numpy import array, random
from digitalseq_b2g import digitalseq_b2g
#from python_prototype.distribution.DigitalSeq import DigitalSeq as digitalseq_b2g

m = 3 # generate 2**3 points
s = 2 # in 2 dimensions

# straightforward enumeration of sobol points
# using generating matrices from file "sobol_Cs.col": (This takes some, time this is a big file!)
gen = digitalseq_b2g("sobol_Cs.col", m=m, s=s)
print("== Unshifted points")
for x in gen:
    print(x)
# let's now print these points as integers (ie binary vectors)
gen.reset()
print("== Unshifted points as integers")
for x in gen:
    print(gen.cur)

# now using digital shifting
gen.reset() # just reset the generator, or initialize it again as above
#gen = digitalseq_b2g("sobol_Cs.col", m=m, s=s)
M = 2 # number of random shifts
t = gen.t # this is the power of 2 at which the integers are shifted inside the generator
t = max(32, t) # we will guarantee at least a depth of 32 bits for the shift
ct = max(0, t - gen.t) # this is the correction factor to scale the integers
random.seed(0) # use same random numbers each time (for debugging and reproducibility)
shifts = random.randint(2**t, size=(M,s)) # generate random shifts
shifts[0,:] = 0 # set zero shift as first shift (to check output)
# now generate the points and shift them
for shift in shifts:
    print("== Digitally shifted with shift = ", shift)
    for x in gen:
        cur = array(gen.cur) # the points as integers (ie binary vectors)
        cur_shifted = shift ^ (cur * 2**ct) # this works since shift is a numpy array
        x = cur_shifted / 2.**t
        print(x)