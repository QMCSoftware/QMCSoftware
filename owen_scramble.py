""" Binary Tree used to generate samples for Nested Uniform Scrambling (Owen Scrambling) in DigitalNetB2 """

from numpy import * 
class Node:
    def __init__(self,rbits,xb,left,right):
        self.rbits = rbits
        self.xb = xb 
        self.left = left 
        self.right = right

def get_scramble_scalar(xb,t,scramble,rng):
    """
    Args:
        xb (uint64): t-bit integer represtation of bits
        t (int64): number of bits in xb 
        scramble (Node): node in the binary tree 
        rng: random number generator (will be part of the DiscreteDistribution class)
    Example: t = 3, xb = 6 = 110_2, x = .110_2 = .75 
    """
    if scramble.xb is None: # branch node, scramble.rbits is 0 or 1
        r1 = lshift(scramble.rbits,(t-1))
        b = (rshift(xb,(t-1)))&1
        onesmask = 2**(t-1)-1
        xbnext = xb&onesmask
        if (not b) and (scramble.left is None):
            rbits = int(rng.integers(0,onesmask+1))
            scramble.left = Node(rbits,xbnext,None,None)
            return r1+rbits
        elif b and (scramble.right is None):
            rbits = int(rng.integers(0,onesmask+1))
            scramble.right = Node(rbits,xbnext,None,None)
            return r1+rbits
        scramble = scramble.left if b==0 else scramble.right
        return  r1 + get_scramble_scalar(xbnext,t-1,scramble,rng)
    elif scramble.xb != xb: # unseen leaf node
        ogsrbits,orsxb = scramble.rbits,scramble.xb
        b,ubit = None,None
        rmask = 2**t-1
        while True:
            b,ubit,rbit = (rshift(xb,(t-1)))&1,(rshift(orsxb,(t-1)))&1,(rshift(ogsrbits,(t-1)))&1
            scramble.rbits,scramble.xb = rbit,None
            if ubit != b: break
            if b==0: 
                scramble.left = Node(None,None,None,None)
                scramble = scramble.left 
            else:
                scramble.right = Node(None,None,None,None)
                scramble = scramble.right 
            t -= 1
        onesmask = 2**(t-1)-1
        newrbits = int(rng.integers(0,onesmask+1)) 
        scramble.left = Node(newrbits,xb&onesmask,None,None) if b==0 else Node(ogsrbits&onesmask,orsxb&onesmask,None,None)
        scramble.right = Node(newrbits,xb&onesmask,None,None) if b==1 else Node(ogsrbits&onesmask,orsxb&onesmask,None,None)
        rmask ^= onesmask
        return (ogsrbits&rmask)+newrbits
    else: # scramble.xb == xb
        return scramble.rbits # seen leaf node 

def rshift(x,v):
    if v >= 0:
        return x >> v
    else:
        return lshift(x,-v)

def lshift(x,v):
    if v >= 0:
        return x << v
    else:
        return rshift(x,-v)
