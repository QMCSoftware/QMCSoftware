''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
import numpy as np

from algorithms.function.integrand_base import IntegrandBase


class AsianCallFun(IntegrandBase):
    """
    Specify and generate payoff values of an Asian Call option

    TODO - Come up with better, more interpretable, lowercase names for these parameters

    """
    def __init__(self, bm_measure=None, volatility=.5, S0=30, K=25, nominal_value=None):
        super().__init__(nominal_value)
        self.bm_measure = bm_measure
        self.volatility = volatility
        self.S0 = S0
        self.K = K
        self.dimFac = 0

        if self.bm_measure:
            num_bm = len(bm_measure)
            self.fun_list = [AsianCallFun() for i in range(num_bm)]
            self[0].bm_measure = self.bm_measure[0]
            self[0].dimFac = 0
            self[0].dimension = self.bm_measure[0].dimension
            for ii in range(1, num_bm):
                self[ii].BMmeasure = self.bm_measure[ii]
                self[ii].dimFac = self.bm_measure[ii].dimension / self.bm_measure[ii-1].dimension
                self[ii].dimension = self.bm_measure[ii].dimension

    # It looks like coords_in_sequence is not being used here ... is that correct?
    # TODO - This is failing the test_integrate ... what's supposed to happen if self.bm_measure is None?
    def g(self, x, coords_in_sequence):
        v = self.volatility
        d = self.dimension
        s_fine = self.S0 * np.exp((-v ** 2 / 2) * self.bm_measure.measureData['timeVector'] + v * x)
        avg_fine = (self.S0 / 2 + np.sum(s_fine[:, :d - 1], axis=1) + s_fine[:, d - 1] / 2) / d
        y = np.fmax(avg_fine - self.K, 0)

        if self.dimFac > 0:
            s_course = s_fine[:, int(self.dimFac - 1)::int(self.dimFac)]
            d_course = d / self.dimFac
            avg_course = (
                self.S0 / 2 + np.sum(s_course[:, :int(d_course) - 1], axis=1) + s_course[:, int(d_course) - 1] / 2
            ) / d_course
            y -= np.fmax(avg_course - self.K, 0)

        return y
