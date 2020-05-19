"""
Definition of ImportanceSampling, a concrete implementation of TrueMeasure
    m(x) is pdf of measure we do not know how to generate from (mystery)
    k(x) is pdf of measure we can generate discrete distribution samples from (known)
"""

from ._true_measure import TrueMeasure
from ..util import TransformError
from numpy import apply_along_axis

class ImportanceSampling(TrueMeasure):
    """
    Perform importance sampling
    on samples from the discrete distribution
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
        super().__init__()

    def transform_g_to_f(self, g):
        """
        Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution samples. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        def f(samples, *args, **kwargs):
            samples_k = self.k_sample_tf(samples) # transform standard samples to mimic measure_to_sample_from
            md = apply_along_axis(self.m,1,samples_k).squeeze() # pdf of objective measure
            kd = apply_along_axis(self.k,1,samples_k).squeeze() # pdf of sampling measure
            w = md/kd
            g_vals = g(samples_k, *args, **kwargs).squeeze() # evaluations of original function
            return w*g_vals
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        raise Exception("Cannot generate samples mimicking an importance sampling measure")
        