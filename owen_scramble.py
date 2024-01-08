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
"""
if __name__ == "__main__":
    rng = random.default_rng(17)
    t = 4
    xbs = [
        0, # 0000_2 -> 0
        8, # 1000_2 -> 1/2
        4, # 0100_2 -> 1/4
        12, # 1100_2 -> 3/4 = 1/2+1/4
        2, # 0010_2 -> 1/8
        6, # 0110_2 -> 3/8 = 1/4+1/8
        10, # 1010_2 -> 5/8 = 1/2+1/8
        14, # 1110_2 -> 7/8 = 1/2+1/4+1/8
    ]

    r1 = int(rng.integers(0,2))<<(t-1)
    rbitsleft,rbitsright = r1+int(rng.integers(0,2**(t-1))),r1+int(rng.integers(0,2**(t-1)))
    root_node = Node(None,None,Node(rbitsleft,0,None,None),Node(rbitsright,2**(t-1),None,None))

    n = len(xbs)
    xbrs = zeros(n)
    for i in range(n):
        xb = xbs[i]
        b = xb>>(t-1)&1
        first_node = root_node.left if b==0 else root_node.right
        xbr = xb ^ get_scramble_scalar(xb,t,first_node,rng)
        xbrs[i] = xbr
        print("%-7d %-7d"%(xb,xbr))
"""
