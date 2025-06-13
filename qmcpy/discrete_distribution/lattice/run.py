from lattice import Lattice
import numpy as np

# Single replication examples
print("Example 1:")
l = Lattice(2, seed=7)
print(l.gen_samples(4))
print(l.gen_samples(1))
print(l)

print("\nExample 2:")
print(Lattice(dimension=2, randomize=False, order='natural').gen_samples(4, warn=False))

print("\nExample 3:")
print(Lattice(dimension=2, randomize=False, order='linear').gen_samples(4, warn=False))

print("\nExample 4:")
print(Lattice(dimension=2, randomize=False, order='gray').gen_samples(4, warn=False))

print("\nExample 5:")
l = Lattice(2, generating_vector=25, seed=55)
print(l.gen_samples(4))
print(l)

print("\nExample 6:")
print(Lattice(dimension=4, randomize=False, seed=353, generating_vector=26).gen_samples(8, warn=False))

print("\nExample 7:")
print(Lattice(dimension=3, randomize=False, generating_vector="LDData/main/lattice/mps.exod2_base2_m20_CKN.txt").gen_samples(8, warn=False))

# Multiple replications examples
print("\nExample 8:")
print(Lattice(3, seed=7, replications=2).gen_samples(4))

print("\nExample 9:")
print(Lattice(3, seed=7, generating_vector=25, replications=2).gen_samples(4))
