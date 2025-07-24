from qmcpy import *
from qmcpy.util import *
import numpy as np
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
            BernoulliCont(DigitalNetB2(d)),
            BernoulliCont(DigitalNetB2(d),lam=[.25,.75]),
            SciPyWrapper(DigitalNetB2(2),[scipy.stats.triang(c=.1),scipy.stats.uniform(loc=1,scale=2)]),
            SciPyWrapper(DigitalNetB2(2),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            for _tm in [tm]+tm.spawn(1):
                t = _tm.gen_samples(4)
                self.assertTrue(t.shape==(4,2))
                self.assertTrue(t.dtype==np.float64)
                x = _tm.discrete_distrib.gen_samples(4)
                xtf,jtf = _tm._jacobian_transform_r(x,return_weights=True)
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
            BernoulliCont(DigitalNetB2(d)),
            SciPyWrapper(DigitalNetB2(2),scipy.stats.triang(c=.1,loc=1,scale=2)),
        ]
        for tm in tms:
            s = 3
            for spawn_dim in [4,[1,4,6]]:
                spawns = tm.spawn(s=s,dimensions=spawn_dim)
                self.assertTrue(len(spawns)==s)
                self.assertTrue(all(type(spawn)==type(tm) for spawn in spawns))
                self.assertTrue((np.array([spawn.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.transform.transform.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((np.array([spawn.discrete_distrib.d for spawn in spawns])==spawn_dim).all())
                self.assertTrue((all(spawn.discrete_distrib!=tm.discrete_distrib for spawn in spawns)))
                self.assertTrue(all(spawn.transform!=tm.transform for spawn in spawns))

class TestMatern(unittest.TestCase):
    def test_sklearn_equivalence(self):
        points = np.array([[5, 4], [1, 2], [0, 0]])
        mean = np.full(3, 1.1)
        
        m2 = MaternGP(Lattice(dimension = 3,seed=7), points, length_scale = 4, nu = 2.5, variance = 0.01, mean=mean, nugget=1e-6)
        from sklearn import gaussian_process as gp  #checking against scikit's Matern
        kernel2 = gp.kernels.Matern(length_scale = 4, nu=2.5)
        cov2 = 0.01 * kernel2.__call__(points) + 1e-6*np.eye(m2.covariance.shape[-1])
        assert np.allclose(cov2, m2.covariance)

if __name__ == "__main__":
    unittest.main()
