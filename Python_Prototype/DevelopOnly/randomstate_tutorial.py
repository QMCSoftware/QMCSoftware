# http://bashtage.github.io/ng-numpy-randomstate/doc/index.html


from randomstate.entropy import random_entropy
import randomstate.prng.pcg64 as pcg64
entropy = random_entropy(4)
from functools import reduce
seed = reduce(lambda x, y: x + y, [long(entropy[i]) * 2 ** (32 * i) for i in range(4)])
streams = [pcg64.RandomState(seed, stream) for stream in range(10)]

from randomstate.entropy import random_entropy
import randomstate.prng.xorshift1024 as xorshift1024
import numpy as np
entropy = random_entropy(2).astype(np.uint64)
# 64-bit number as a seed
seed = entropy[0] * 2**32 + entropy[1]
blocks = []
for i in range(10):
    block = xorshift1024.RandomState(seed)
    block.jump(i)
    blocks.append(block)


import randomstate.prng.pcg64 as pcg64
rs = pcg64.RandomState()
rs_gen = pcg64.RandomState()
rs_gen.set_state(rs.get_state())

advance = 2**127
rngs = [rs]
for _ in range(9):
    rs_gen.advance(advance)
    rng = pcg64.RandomState()
    rng.set_state(rs_gen.get_state())
    rngs.append(rng)