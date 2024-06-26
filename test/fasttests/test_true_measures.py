from qmcpy import *
from qmcpy.util import *
from numpy import *
import scipy.stats
import unittest


class TestTrueMeasure(unittest.TestCase):
    """ General tests for TrueMeasures """
    
    def test_abstract_methods(self):
        d = 2
        tms = [
            Uniform(DigitalNetB2(d)),
            Uniform(DigitalNetB2(d),lower_bound=[1,2],upper_bound=[2,3]),
            Kumaraswamy(DigitalNetB2(d)),
            Kumaraswamy(DigitalNetB2(d),a=[2,4],b=[1,3]),
            JohnsonsSU(DigitalNetB2(d)),
            JohnsonsSU(DigitalNetB2(d),gamma=[1,2],xi=[4,5],delta=[7,8],lam=[10,11]),
            Gaussian(DigitalNetB2(d)),
            Gaussian(DigitalNetB2(d),mean=[1,2],covariance=[[9,5],[5,9]],decomp_type='Cholesky'),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d)))),
            BrownianMotion(DigitalNetB2(d)),
            BrownianMotion(DigitalNetB2(d),t_final=2,drift=3,decomp_type='Cholesky'),
            GeometricBrownianMotion(DigitalNetB2(d)),
            GeometricBrownianMotion(DigitalNetB2(d),t_final=2,drift=3,decomp_type='Cholesky'),
            BernoulliCont(DigitalNetB2(d)),
            BernoulliCont(DigitalNetB2(d),lam=[.25,.75]),
            SciPyWrapper(DigitalNetB2(2),[scipy.stats.triang(c=.1),scipy.stats.uniform(loc=1,scale=2)]),
            SciPyWrapper(DigitalNetB2(2),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            for _tm in [tm]+tm.spawn(1):
                t = _tm.gen_samples(4)
                self.assertTrue(t.shape==(4,2))
                self.assertTrue(t.dtype==float64)
                x = _tm.discrete_distrib.gen_samples(4)
                xtf,jtf = _tm._jacobian_transform_r(x)
                self.assertTrue(xtf.shape==(4,d),jtf.shape==(4,))
                w = _tm._weight(x)
                self.assertTrue(w.shape==(4,))
                s = str(_tm)
            
    def test_spawn(self):
        d = 3
        tms = [
            Uniform(DigitalNetB2(d)),
            Lebesgue(Uniform(DigitalNetB2(d))),
            Lebesgue(Gaussian(DigitalNetB2(d))),
            Kumaraswamy(DigitalNetB2(d)),
            JohnsonsSU(DigitalNetB2(d)),
            Gaussian(DigitalNetB2(d)),
            Gaussian(Kumaraswamy(Kumaraswamy(DigitalNetB2(d)))),
            BrownianMotion(DigitalNetB2(d)),
            GeometricBrownianMotion(DigitalNetB2(d)),
            BernoulliCont(DigitalNetB2(d)),
            SciPyWrapper(DigitalNetB2(2),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            s = 3
            for spawn_dim in [4,[1,4,6]]:
                spawns = tm.spawn(s=s,dimensions=spawn_dim)
                self.assertTrue(len(spawns)==s)
                self.assertTrue(all(type(spawn)==type(tm) for spawn in spawns))
                self.assertTrue((array([spawn.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((array([spawn.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((array([spawn.transform.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((array([spawn.discrete_distrib.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((all(spawn.discrete_distrib!=tm.discrete_distrib for spawn in spawns)))
                self.assertTrue(all(spawn.transform!=tm.transform for spawn in spawns))


class TestGeometricBrownianMotion(unittest.TestCase):

    def setUp(self):
        """
        Sets up a sampler and the initial conditions for the tests in TestGeometricBrownianMotion.

        The sampler is initialized with a dimension of 4 and a seed of 7.

        The GeometricBrownianMotion instance is initialized with the sampler, a final time of 2, a drift of 0.1,
        and a diffusion of 0.2.
        """
        self.sampler = DigitalNetB2(4, seed=7)
        self.gbm = GeometricBrownianMotion(self.sampler, t_final=2, drift=0.1, diffusion=0.2)

    def test_init(self):
        """
        Tests the initialization of the GeometricBrownianMotion instance.

        The test checks several aspects of the GeometricBrownianMotion instance:
        - The final time (t_final) is correctly set to 2.
        - The drift is correctly set to 0.1.
        - The diffusion is correctly set to 0.2.
        - The range is correctly set to an array from negative infinity to positive infinity.
        """
        self.assertEqual(self.gbm.t, 2)
        self.assertEqual(self.gbm.drift, 0.1)
        self.assertEqual(self.gbm.diffusion, 0.2)
        self.assertTrue(allclose(self.gbm.range, array([[-inf, inf]])))

    def test_init_error(self):
        """
        Tests the initialization of the GeometricBrownianMotion instance with invalid parameters.

        The test checks several scenarios where the initialization should fail:
        - The final time (t_final) is set to a negative value.
        - The diffusion is set to a negative value.
        - The initial value is set to a negative value.
        - The decomposition type (decomp_type) is not 'PCA' or 'Cholesky'.

        In each of these scenarios, the initialization is expected to raise a ValueError.
        """
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(self.sampler, t_final=-1, drift=0.1, diffusion=0.2)
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(self.sampler, t_final=2, drift=0.1, diffusion=-0.2)
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(self.sampler, t_final=2, drift=0.1, diffusion=0.2, initial_value=-1)
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(self.sampler, t_final=2, drift=0.1, diffusion=0.2, decomp_type='InvalidType')


if __name__ == "__main__":
    unittest.main()