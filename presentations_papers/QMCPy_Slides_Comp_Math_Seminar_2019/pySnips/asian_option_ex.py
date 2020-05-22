tm = BrownianMotion(dimension = [4, 16, 64], \
        time_vector = \ 
            [
                arange(1/4, 5/4, 1/4),
                arange(1/16, 17/16, 1/16),
                arange(1/64, 65/64, 1/64)
            ])
integrand = \ 
    AsianCall(tm, 
        volatility = .5, \
        start_price = 30, \
        strike_price = 25, \ 
        interest_rate = .01, \
        mean_type = 'arithmetic')
dd = IIDStdGaussian(rng_seed = 7)
stop = CLT(dd, tm, abs_tol=.05)
sol, data = integrate(integrand, tm, dd, stop)