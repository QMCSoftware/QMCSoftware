class TestCustomIIDDistribution(unittest.TestCase):
    """ Unit tests for CustomIIDDistribution DiscreteDistribution. """

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        distribution.gen_samples(10)

    def test_set_dimension(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        self.assertRaises(DimensionError,distribution.set_dimension,3)
                

class TestAcceptanceRejectionSampling(unittest.TestCase):
    """ Unit tests for AcceptanceRejectionSampling DiscreteDistribution. """

    def test_gen_samples(self):
        def f(x):
            # see sampling measures demo
            x = x if x<.5 else 1-x 
            density = 16*x/3 if x<1/4 else 4/3
            return density  
        distribution = AcceptanceRejectionSampling(
            objective_pdf = f,
            measure_to_sample_from = Uniform(IIDStdUniform(1)))
        distribution.gen_samples(10)
    
    def test_set_dimension(self):
        distribution = AcceptanceRejectionSampling(lambda x: 1, Uniform(IIDStdGaussian(2)))
        self.assertRaises(DimensionError,distribution.set_dimension,3)


class TestInverseCDFSampling(unittest.TestCase):
    """ Unit tests for InverseCDFSampling DiscreteDistribution. """

    def test_gen_samples(self):
        distribution = InverseCDFSampling(Lattice(2),
            inverse_cdf_fun = lambda u,l=5: -log(1-u)/l)
                        # see sampling measures demo
        distribution.gen_samples(8)
    
    def test_set_dimension(self):
        distribution = InverseCDFSampling(Lattice(2),lambda u: u)
        self.assertRaises(DimensionError,distribution.set_dimension,3)


class TestIdentitalToDiscrete(unittest.TestCase):
    """ Unit tests for IdentitalToDiscrete Measure. """

    def test_gen_samples(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentitalToDiscrete(distribution)
        samples = measure.gen_samples(n=5)
    
    def test__transform_g_to_f(self):
        # implicitly called from Integrand superclass constructor
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentitalToDiscrete(distribution)
        l = Linear0(measure)
        l.f(distribution.gen_samples(2**3))
    
    def test_set_dimension(self):
        distribution = CustomIIDDistribution(lambda n: random.poisson(lam=5,size=(n,2)))
        measure = IdentitalToDiscrete(distribution)
        self.assertRaises(DimensionError,measure.set_dimension,3)


class TestImportanceSampling(unittest.TestCase):
    """ Unit tests for ImportanceSampling Measure. """
    
    def test_construct(self):
        def quarter_circle_uniform_pdf(x):
            # see sampling measures demo
            x1,x2 = x
            if sqrt(x1**2+x2**2)<1 and x1>=0 and x2>=0:
                return 4/pi 
            else:
                return 0. # outside of quarter circle
        measure = ImportanceSampling(
            objective_pdf = quarter_circle_uniform_pdf,
            measure_to_sample_from = Uniform(Lattice(dimension=2,seed=9)))