# Generate X = [x1,x2] for right plot
tm = Gaussian(dimension=[2], \
              mean=[[3,7]], variance=[[9,9]])
dd = Sobol(rng_seed = 7)
tm.set_tm_gen(dd) # Initialize below method
X = tm.gen_tm_samples(r=1, n=128).squeeze()