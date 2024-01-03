from qmcpy import *
import numpy as np 

class Node:
    def __init__(self,rbits,xb,left,right):
        self.rbits = rbits
        self.xb = xb 
        self.left = left 
        self.right = right

d = 3
dnb2 = DigitalNetB2(3,randomize=False)

new_seeds = dnb2._base_seed.spawn(d)
rngs = [np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(d)]
print(rngs)

root_nodes = [None]*d
for j in range(d):
    r1 = int(rngs[j].integers(0,2))<<(dnb2.t_lms-1)
    rbitsleft,rbitsright = r1+int(rngs[j].integers(0,2**(dnb2.t_lms-1))),r1+int(rngs[j].integers(0,2**(dnb2.t_lms-1)))
    root_nodes[j] = Node(None,None,Node(rbitsleft,0,None,None),Node(rbitsright,2**(dnb2.t_lms-1),None,None))
print(root_nodes)