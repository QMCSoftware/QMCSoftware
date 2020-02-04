""" Definition of MeanVarData, a concrete implementation of AccumData """

from ._accum_data import AccumData
from time import process_time
from numpy import array, finfo, float32, full, inf, nan, tile, zeros

EPS = finfo(float32).eps


class MeanVarData(AccumData):
    """
    Accumulated data for IIDDistribution calculations,
    and store the sample mean and variance of integrand values
    """

    parameters = ['levels','solution','n','n_total','confid_int']

    def __init__(self, levels, n_init):
        """
        Initialize data instance

        Args:
            levels (int): number of integrands
            n_init (int): initial number of samples
        """
        self.levels = levels
        self.solution = nan
        self.muhat = full(self.levels, inf)  # sample mean
        self.sighat = full(self.levels, inf)  # sample standard deviation
        self.t_eval = zeros(self.levels)  # processing time for each integrand
        self.n = tile(n_init, self.levels) # currnet number of samples
        self.n_total = 0  # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        super().__init__()

    def update_data(self, integrands, measures):
        """
        Update data

        Args:
            integrands (Integrand): an instance of Integrand
            measures (Measure): an instance of Measure

        Returns:
            None
        """
        if self.levels == 1:
            # single level -> make args appear multilevel
            integrands = [integrands]
            measures = [measures]
        for l in range(self.levels):
            t_start = process_time()  # time the integrand values
            samples = measures[l].gen_samples(n=self.n[i])
            y = integrand[l].f(samples).squeeze()
            self.t_eval[l] = max(process_time() - t_start, EPS)
            self.sighat[l] = y.std()  # compute the sample standard deviation
            self.muhat[l] = y.mean()  # compute the sample mean
            self.n_total += self.n[i]  # add to total samples
        self.solution = self.muhat.sum()  # tentative solution
