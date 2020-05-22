# Generate X = [x1,x2] for left plot
dd = IIDStdUniform(rng_seed = 7)
X = dd.gen_dd_samples(1, 128, 2).squeeze()