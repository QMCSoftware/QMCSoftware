from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np

class ErrorApproximation(AbstractIntegrand):
    def __init__(self, integrand):
        r"""
        Args:
            integrand (AbstractIntegrand): Integrand to use in error approximation
        """
        self.integrand = integrand
        # Use DigitalNetB2 with same dimension as integrand
        self.d = getattr(integrand, 'd', None)
        if self.d is None:
            raise ParameterError("Integrand must have attribute 'd' for dimension.")
        self.true_integrand = type(integrand)(DigitalNetB2(dimension=self.d, seed=7))
        self.true_value = self.true_integrand(2**21).mean()


    def deterministic_error(self, n, distrib=None):
        """
        Returns the absolute and relative error for n samples.
        User can specify the distribution object; defaults to DigitalNetB2.
        """
        if distrib is None:
            distrib = DigitalNetB2(dimension=self.d, seed=7)
        x = distrib.gen_samples(n)
        approx = self.integrand.f(x).mean()
        abs_err = np.abs(approx - self.true_value)
        rel_err = abs_err / np.abs(self.true_value)
        return abs_err, rel_err
    
    def mean_abs_error(self, n, repetitions=30, distrib=None):
        """
        Returns the mean absolute error over a number of repetitions, using random shift for each replication.
        User can specify the distribution object; defaults to DigitalNetB2.
        """
        if distrib is None:
            distrib = DigitalNetB2(dimension=self.d, seed=7)
        x_base = distrib.gen_samples(n)
        errors = []
        for _ in range(repetitions):
            shift = np.random.uniform(0, 1, self.d)
            x_shifted = (x_base + shift) % 1
            approx = self.integrand.f(x_shifted).mean()
            abs_err = np.abs(approx - self.true_value)
            errors.append(abs_err)
        return np.mean(errors)
    
    def mean_sqr_error(self, n, repetitions=30, distrib=None):
        """
        Returns the mean square error over a number of repetitions, using random shift for each replication.
        User can specify the distribution object; defaults to DigitalNetB2.
        """
        if distrib is None:
            distrib = DigitalNetB2(dimension=self.d, seed=7)
        x_base = distrib.gen_samples(n)
        errors = []
        for _ in range(repetitions):
            shift = np.random.uniform(0, 1, self.d)
            x_shifted = (x_base + shift) % 1
            approx = self.integrand.f(x_shifted).mean()
            sqr_err = (approx - self.true_value)**2
            errors.append(sqr_err)
        return np.mean(errors)
