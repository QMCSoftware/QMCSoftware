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
            BernoulliCont(DigitalNetB2(d)),
            BernoulliCont(DigitalNetB2(d),lam=[.25,.75]),
            SciPyWrapper(DigitalNetB2(2),scipy.stats.triang,c=[0.1,.2],loc=[1,2],scale=[3,4]),
            SciPyWrapper(SciPyWrapper(DigitalNetB2(2),scipy.stats.triang,c=[0.1,.2]),scipy.stats.triang,c=[.3,.4]),
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
            BernoulliCont(DigitalNetB2(d)),
            SciPyWrapper(SciPyWrapper(DigitalNetB2(2),scipy.stats.triang,c=.25),scipy.stats.triang,c=.75),
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


if __name__ == "__main__":
    unittest.main()
