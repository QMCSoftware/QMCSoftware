from ._stopping_criterion import StoppingCriterion
from ..discrete_distribution import Kronecker
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from scipy.linalg import solve_toeplitz
from scipy.sparse import identity
import time
import numpy as np

# 1. Generate samples starting at some n_min.
# 2. Calculate the error using equation (10b)
# 3. If 2*error is not less than the abs_tol set by the user, then double (or increase in some way) the number of points for the next iteration. 
# 4. Then, repeat steps 1-3 if necessary.
# 5. Once the tolerance is met, use equation (16) to calculate the estimated integral.

class CubBayesKronecker(StoppingCriterion):
    # Based on cub_qmc_bayes_lattice_g.py
    def __init__(self, integrand, abs_tol = 1e-2, n_init = 2 ** 8, n_max = int(1e6)):
        # Set Attributes
        self.abs_tol = float(abs_tol)
        self.n_init = n_init
        self.n_max = n_max

        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure 
        self.discrete_distrib = self.integrand.discrete_distrib

        # Verify Compliant Construction
        super(CubBayesKronecker, self).__init__(allowed_levels=['single'], allowed_distribs=[Kronecker], allow_vectorized_integrals=True)

    def integrate(self):
        # n = self.n_init
        n = 4096
        invert_time = 0
        start = time.time()
        while True:
            
            self.ones_vector = np.ones((n, 1))
            self.ones_transpose = self.ones_vector.T

            samples = self.discrete_distrib.gen_samples(n + 1)
            observed = self.integrand.f(samples[1:], periodization_transform = 'BAKER')

            invert_start = time.time()
            inverted_gram = self._invert_gram(n, samples[:-1])
            invert_end = time.time()
            invert_time += invert_end - invert_start
            print(invert_time)
            s = np.sqrt(self.calculate_variance(n, observed, inverted_gram))
            error = self.calculate_error_CI(s, inverted_gram)
        
            if 2 * error < self.abs_tol:
                end = time.time()
                return (self.approximate_integral(n, observed), (n, end - start, invert_time))
            
            elif n >= self.n_max:
                print('Already used maximum allowed sample size %d.')
                break
            else:
                n *= 2
            

    # Equation (10b)
    def calculate_error_CI(self, s, inverted_gram):
        return 2.58 * s * np.sqrt(1 - self.ones_transpose @ inverted_gram @ self.ones_vector)

    
    # Using equation (12)
    def calculate_variance(self, n, y, inverted_gram):
        # equivalent to inverted_gram @ ones_vector @ ones_transpose @ inverted_gram
        expr = np.sum(inverted_gram, axis = 1, keepdims = True) @ np.sum(inverted_gram, axis = 0, keepdims = True)
        total = np.sum(inverted_gram)

        return (1 / n) * y.T @ (inverted_gram - (expr / total)) @ y

    
    def approximate_integral(self, n, observed):
        return (1 / n) * np.sum(observed)
    

    def _invert_gram(self, n, samples):
        # First row would be i = 1, 2, ..., n and j = 1
        # So, (i - j) * alpha would be (0, 1, ..., n - 1) * alpha

        # First row and column are the same
        first_row = k_tilde(samples)
        return solve_toeplitz((first_row, first_row), identity(n).toarray(), check_finite = False)


def k_tilde(x):
    return np.prod(1 + (x * (x - 1) + 1/6), axis = 1)

