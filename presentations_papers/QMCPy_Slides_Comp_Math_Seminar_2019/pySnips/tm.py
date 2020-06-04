# Generate X = [x1,x2] for right-most plot
tm = Uniform(dimension = 2)
dd = Sobol(rng_seed = 7)
tm.set_tm_gen(dd) # Initialize below method
X = tm.gen_tm_samples(r=1, n=128).squeeze()