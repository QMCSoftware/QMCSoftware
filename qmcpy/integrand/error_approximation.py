from .genz import Genz
from .keister import Keister
from ..discrete_distribution import DigitalNetB2, AbstractDiscreteDistribution
from .other_qmc_integrals import QMCIntegrals
from ..util import ParameterError
import numpy as np

class ErrorApproximation(AbstractDiscreteDistribution): #Will add functionality for point sets later
    r"""
    
    """

    def __init__(self, sampler, n=[2**10]): #add an argument to specify which sample set to compare against, default to Sobol w/ 2e6 points
        r"""
        Args:
            sampler (AbstractDiscreteDistribution): Point set or sequence to use in error approximation
            n (list): For sequences only. Optional. If sampler is a sequence, specifies which sample sizes to use.
                - Defaults to [2**10]
        """
        self.sampler = sampler
        self.d = getattr(sampler, 'd', None)
        if self.d is None:
            raise ParameterError("Point set must have attribute 'd' for dimension.")
        #This will change when point set functionality is added, since n is already determined for them
        self.n = n
        #More valid integrands to be added later
        self.valid_integrands = ['genz_oscillatory', 'genz_corner_peak', 'ra_sum', 'ra_prod', 'ra_sin', 'keister']
        self.true_values = {}
        for integrand_type in self.valid_integrands:
            if integrand_type == 'genz_oscillatory':
                kind_func = 'OSCILLATORY'
                integrand = Genz(DigitalNetB2(dimension=self.d, seed=7), kind_func=kind_func, kind_coeff=2)
                self.true_values[integrand_type] = integrand(2**21).mean()
            elif integrand_type == 'genz_corner_peak':
                kind_func = 'CORNER PEAK'
                integrand = Genz(DigitalNetB2(dimension=self.d, seed=7), kind_func=kind_func, kind_coeff=2)
                self.true_values[integrand_type] = integrand(2**21).mean()
            elif integrand_type == 'ra_sum' or integrand_type == 'ra_prod' or integrand_type == 'ra_sin':
                self.true_values[integrand_type] = 1.0
            elif integrand_type == 'keister':
                integrand = Keister(DigitalNetB2(dimension=self.d, seed=7))
                self.true_values[integrand_type] = integrand(2**21).mean()
            #More integrands to be added later
        
    def error_approx(self, err_type, integrands, repetitions=30, genz_coeff=2):
        r"""
        Args:
            err_type (str): Type of error approximation to compute. Options are:
                - 'abs': Absolute error for n samples.
                - 'rel': Relative error for n samples.
                - 'mae': Mean absolute error over repetitions with random shifts.
                - 'rmse': Root mean square error over repetitions with random shifts.
            integrands (list): List of integrand objects to evaluate. Options are:
                - 'all': All integrands available
                - 'genz': All Genz integrands
                    - 'genz_oscillatory': Genz oscillatory integrand
                    - 'genz_corner_peak': Genz corner peak integrand
                - 'ra': All Roos-Arnold integrands
                    - 'ra_sum': Roos-Arnold sum integrand
                    - 'ra_prod': Roos-Arnold product integrand
                    - 'ra_sin': Roos-Arnold sine integrand
                - 'keister': Keister integrand
            repetitions (int): Optional. Number of random shift repetitions for 'mae' and 'rmse' error types.
                - Defaults to 30.
            genz_coeff (int): Optional. The type of coefficients to use for genz integrands, as described in genz.py
                - Must be 1, 2, or 3, defaults to 2.

        Returns:
            errors (list): List of lists of errors corresponding to each sample size for each integrand.
        """
        #Clean up inputs
        err_type = str(err_type).lower().strip()
        if err_type != 'abs' and err_type != 'rel' and err_type != 'mae' and err_type != 'rmse':
            raise ParameterError("Invalid error type: %s" % err_type)
        if genz_coeff not in [1,2,3]:
            raise ParameterError("Expects genz_coeff in [1,2,3]")
        for i in range(len(integrands)):
            integrands[i] = str(integrands[i]).lower().strip().replace(" ","_").replace("-","_")
        #Get list of requested integrand types
        integrand_types = []
        if 'all' in integrands:
            integrand_types = self.valid_integrands
        else:
            for integrand in integrands:
                if str(integrand) not in integrand_types:
                    if str(integrand) == 'genz':
                        integrand_types.append('genz_oscillatory')
                        integrand_types.append('genz_corner_peak')
                    elif str(integrand) == 'ra':
                        integrand_types.append('ra_sum')
                        integrand_types.append('ra_prod')
                        integrand_types.append('ra_sin')
                    elif integrand in self.valid_integrands:
                        integrand_types.append(str(integrand))
                    else:
                        raise ParameterError("Invalid integrand specified: %s"%str(integrand))
        if len(integrand_types) == 0:
            raise ParameterError("No valid integrands specified.")
        #Compute errors
        err_dict = {}
        for integrand_type in integrand_types:
            if integrand_type == 'genz_oscillatory':
                integrand = Genz(self.sampler, kind_func = 'OSCILLATORY', kind_coeff=genz_coeff)
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            elif integrand_type == 'genz_corner_peak':
                integrand = Genz(self.sampler, kind_func = 'CORNER PEAK', kind_coeff=genz_coeff)
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            elif integrand_type == 'ra_sum':
                integrand = QMCIntegrals(self.sampler, kind_func='RA SUM')
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            elif integrand_type == 'ra_prod':
                integrand = QMCIntegrals(self.sampler, kind_func='RA PROD')
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            elif integrand_type == 'ra_sin':
                integrand = QMCIntegrals(self.sampler, kind_func='RA SIN')
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            elif integrand_type == 'keister':
                integrand = Keister(self.sampler)
                err_dict[str(integrand_type)] = self.err(integrand_type,integrand,repetitions,err_type)
            #More integrands to be added later
        return [err_dict[str(type)] for type in integrand_types]
        
    def err(self, integrand_type, integrand, repetitions, err_type):
        if err_type == 'abs':
            temp_err_list = []
            for n in self.n:
                approx = integrand(n).mean()
                abs_err = np.abs(approx - self.true_values[integrand_type])
                temp_err_list.append(abs_err)
            return temp_err_list
        elif err_type == 'rel':
            temp_err_list = []
            for n in self.n:
                approx = integrand(n).mean()
                rel_err = (approx - self.true_values[integrand_type])/self.true_values[integrand_type]
                temp_err_list.append(rel_err)
            return temp_err_list
        elif err_type == 'mae':
            temp_err_list = []
            for n in self.n:
                rep_errs = []
                for rep in range(repetitions):
                    shift = np.random.uniform(0, 1, self.d)
                    x_shifted = (self.sampler.gen_samples(n) + shift) % 1
                    approx = integrand.f(x_shifted).mean()
                    abs_err = np.abs(approx - self.true_values[integrand_type])
                    rep_errs.append(abs_err)
                mae = np.mean(rep_errs)
                temp_err_list.append(mae)
            return temp_err_list
        elif err_type == 'rmse':
            temp_err = []
            for n in self.n:
                rep_errs = []
                for rep in range(repetitions):
                    shift = np.random.uniform(0, 1, self.d)
                    x_shifted = (self.sampler.gen_samples(n) + shift) % 1
                    approx = integrand.f(x_shifted).mean()
                    sqr_err = (approx - self.true_values[integrand_type])**2
                    rep_errs.append(sqr_err)
                rmse = np.sqrt(np.mean(np.array(rep_errs)))
                temp_err.append(rmse)
            return temp_err
                        
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
    
    def root_mean_sqr_error(self, n, repetitions=30, distrib=None):
        """
        Returns the root mean square error over a number of repetitions, using random shift for each replication.
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
        return np.sqrt(np.mean(errors))
