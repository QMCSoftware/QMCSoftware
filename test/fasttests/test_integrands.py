from qmcpy import *
from qmcpy.util import *
from numpy import *
import unittest
import scipy.stats

class TestIntegrand(unittest.TestCase):
    """ General tests for Integrand """

    def test_abstract_methods(self):
        d = 2
        integrands = [
            AsianOption(DigitalNetB2(d),call_put='call',mean_type='arithmetic'),
            AsianOption(DigitalNetB2(d),call_put='put',mean_type='arithmetic'),
            AsianOption(DigitalNetB2(d),call_put='call',mean_type='geometric'),
            AsianOption(DigitalNetB2(d),call_put='put',mean_type='geometric'),
            BoxIntegral(DigitalNetB2(d),s=1),
            BoxIntegral(DigitalNetB2(d),s=[3,5,7]),
            CustomFun(Uniform(DigitalNetB2(d)),lambda x: x.prod(1)),
            CustomFun(Uniform(Kumaraswamy(SciPyWrapper(DigitalNetB2(d),scipy.stats.triang,c=[0.1,.2]))),lambda x: x.prod(1)),
            CustomFun(Gaussian(DigitalNetB2(2)),lambda x: x,dprime=d),
            EuropeanOption(DigitalNetB2(d),call_put='call'),
            EuropeanOption(DigitalNetB2(d),call_put='put'),
            Keister(DigitalNetB2(d)),
            Keister(Gaussian(DigitalNetB2(d))),
            Keister(BrownianMotion(Kumaraswamy(DigitalNetB2(d)))),
            Linear0(DigitalNetB2(d)),
        ]
        for ao_ml_dim in [[2,4,8],[3,5]]:
            ao_og = AsianOption(DigitalNetB2(d),multilevel_dims=ao_ml_dim)
            ao_spawns = ao_og.spawn(levels=arange(len(ao_ml_dim)))
            for ao_spawn,true_d in zip(ao_spawns,ao_ml_dim):
                self.assertTrue(ao_spawn.d==true_d)
            integrands += ao_spawns
            s = str(ao_og)
        for ml_option in [
            MLCallOptions(DigitalNetB2(d),option='european'),
            MLCallOptions(BrownianMotion(DigitalNetB2(d)),option='asian'),
            ]:
            for levels in [[0,1],[3,5,7]]:
                ml_spawns = ml_option.spawn(levels=levels)
                for ml_spawn,level in zip(ml_spawns,levels):
                    self.assertTrue(ml_spawn.d==ml_spawn._dimension_at_level(level))
                integrands += ml_spawns
            s = str(ml_option)
        n = 8
        spawned_integrands = [integrand.spawn(levels=0)[0] for integrand in integrands]
        for integrand in integrands+spawned_integrands:
            x = integrand.discrete_distrib.gen_samples(n)
            s = str(integrand)
            for ptransform in ['None','Baker','C0','C1','C1sin','C2sin','C3sin']:
                y = integrand.f(x,periodization_transform=ptransform)
                self.assertTrue(y.shape==(n,integrand.dprime))
                self.assertTrue(isfinite(y).all())
                self.assertTrue(y.dtype==float64)


if __name__ == "__main__":
    unittest.main()
