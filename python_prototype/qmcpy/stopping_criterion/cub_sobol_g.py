""" Definition for CubSobol_g, a concrete implementation of StoppingCriterion

Adapted from
    https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubSobol_g.m

Reference:
    
    [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
    Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
    GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
    Available from http://gailgithub.github.io/GAIL_Dev/
"""

from ._stopping_criterion import StoppingCriterion


class CubSobol_g(StoppingCriterion):
    """
    Stopping criterion for Lattice sequence with garunteed accuracy

    Guarantee
        This algorithm computes the integral of real valued functions in :math:`[0,1]^d`
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than :math:`[0,1]^d`, this cone
        condition applies to :math:`f \circ \psi` (the composition of the
        functions) where :math:`\psi` is the transformation function for :math:`[0,1]^d` to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

    def __init__(self, distribution):
        pass

    def stop_yet(self):
        """ Determine when to stop """
        raise Exception("Not yet implemented")
