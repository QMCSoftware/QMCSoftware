from ._true_measure import TrueMeasure
from ..util import TransformError
from . import Uniform
from ..discrete_distribution import Lattice
from numpy import *


class ImportanceSampling(TrueMeasure):
    """
    Importance Sampling. 

    >>> def quarter_circle_uniform_pdf(x):
    ...     x1,x2 = x
    ...     if sqrt(x1**2+x2**2)<1 and x1>=0 and x2>=0:
    ...         return 4./pi # 1./(pi*(1**2)/4)
    ...     else:
    ...         return 0. # outside of quarter circle
    >>> tm = ImportanceSampling(
    ...     objective_pdf = quarter_circle_uniform_pdf,
    ...     measure_to_sample_from = Uniform(Lattice(2,seed=7)))
    """

    parameters = []

    def __init__(self, objective_pdf, measure_to_sample_from):
        """
        Args:
            objective_pdf (function): pdf function of objective measure
            measure_to_sample_from (TrueMeasure): true measure we can sample from
        """
        self.mimics = 'None'
        self.m = objective_pdf
        self.measure = measure_to_sample_from
        self.distribution = self.measure.distribution
        self.dimension = self.distribution.dimension
        if not hasattr(self.measure,'pdf'):
            raise TransformError('measure_to_sample_from must have pdf methd')
        self.k = self.measure.pdf
        if not hasattr(self.measure,'_tf_to_mimic_samples'):
            raise TransformError('measure_to_sample_from must have _tf_to_mimic_samples method')
        self.k_sample_tf = self.measure._tf_to_mimic_samples
        super(ImportanceSampling,self).__init__()

    def _transform_g_to_f(self, g):
        """ See abstract method. """
        def f(samples, *args, **kwargs):
            samples_k = self.k_sample_tf(samples) # transform standard samples to mimic measure_to_sample_from
            md = apply_along_axis(self.m,1,samples_k).squeeze() # pdf of objective measure
            kd = apply_along_axis(self.k,1,samples_k).squeeze() # pdf of sampling measure
            w = md/kd
            g_vals = g(samples_k, *args, **kwargs).squeeze() # evaluations of original function
            return w*g_vals
        return f
        