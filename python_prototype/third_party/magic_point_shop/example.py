from __future__ import print_function
from digitalseq_b2g import digitalseq_b2g

filename = 'sobol_Cs.col' # in the repository this is a link to ../DIGSEQ/sobol_mini.col
seq = digitalseq_b2g(filename)

print("Loaded %s which has number of dimensions s=%d, number of columns m=%d and precision t=%d" 
        % (filename, seq.s, seq.m, seq.t))
print("Bit reversed generating matrices: Csr=%s" % seq.Csr)
print()

def show_state_and_points(n):
    # let's look at the state of the generator:
    print("Current state: previous index k=%d, current point x=%s and cur=%s" % (seq.k, seq.x, seq.cur))
    # print first 10 points:
    for i in range(n):
        seq.next()     # notice always call next first, although the 0th point was already calculated (exceptionally)
        print("%3d %s" % (seq.k, seq.x))
    # look at state again:
    print("Current state: previous index k=%d, current point x=%s and cur=%s" % (seq.k, seq.x, seq.cur))
    print()

print("Freshly constructed sequence; next 10 points:")
show_state_and_points(10)
print("Next 6 points:")
show_state_and_points(6) # the mini sobol only has 16 points

print("Calling reset; next 10 points:")
seq.reset()
show_state_and_points(10)

newk = 4
print("Setting state to k=%d; next 11 points:" % newk)
seq.set_state(newk)
show_state_and_points(12) # the mini sobol only has 16 points
