""" Definition for MeanVarDataRep, a concrete implementation of AccumData """

from ._accum_data import AccumData


class CubatureData(AccumData):
    """
    Accumulated data relavent to cubature algorithms
    """

    def __init__(self, levels, n_init):
        """
        Initialize data instance

        Args:
            levels (int): number of integrands
            n_init (int): initial number of samples
        """
        self.solution = nan
        self.n = tile(n_init, levels).astype(float)  # currnet number of samples
        self.n_total = tile(0, levels)  # total number of samples
        self.confid_int = array([-inf, inf])  # confidence interval for solution
        self.r = 1 # single replication
        super().__init__()

    def update_data(self, integrand, true_measure):
        """
        Update data

        Args:
            integrand (Integrand): an instance of Integrand
            true_measure (TrueMeasure): an instance of TrueMeasure

        Returns:
            None
        """
        for i in range(len(true_measure)):
            n_gen = self.n[i] - self.n_total[i]
            set_x = true_measure[i].gen_tm_samples(self.r, n_gen).squeeze() # n_gen x d samples
            y = integrand[i].f(set_x)
        self.n_total = self.n.copy()  # updated the total evaluations
        # standard deviation of stream means
        self.solution = self.muhat.sum()  # mean of integrand approximations

    def __repr__(self):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return super().__repr__(['r'])
