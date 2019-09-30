# I feel like having a directory named function is just asking for trouble.
# I would consider renaming it to maybe integrands?  Or something more precise (if it makes sense).

# Comments like this are not common -- the record of who did what is in GitHub
""" Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin """

# It is more standard to write np.sqrt than from numpy import sqrt
import numpy as np
from scipy.stats import norm

# NOTE(Mike)
# I am not a fan of direct importing, which is why I changed this function to the way it looks
# It requires a sense of the structure of the code base that shouldn't be necessary during development
# But maybe I just feel that way because I like defining PYTHONPATH each time I run things
# Feel free to do this as you want
from .. import univ_repr


class IntegrandBase(object):
    """
    Specify and generate values $f(\vx)$ for $\vx \in \cx$
        Any sublcass of IntegrandBase must include:
            Methods: g(self, x, coord_index)
    """

    def __init__(self, nominal_value=0):
        # There are a lot of ways to format docstrings ... this is just one I'm familiar with
        # I actually, usually, try not to write them because I think they just make code look busy
        # Generally, I think the best goal is to write self-documenting code, with obvious variable names
        # But if you want to have docstrings, that's totally fine (so long as they stay up to date)
        # They are nice for calling `help` on
        """
        :param nominal_value: $c$, such that $(c, \ldots, c) \in \cx$
        :type nominal_value: float

        """
        # I copied the description of nominal_value, but I do not know what that means
        self.nominal_value = nominal_value
        self.integrand_handle_after_transformation = None
        self.dimension = None

        # This is super awkward and, frankly, I don't think I would recommend this
        # People will have a very difficult time following this structure
        # When you're looking for collaborators, the last thing you want is code with unintuitive design
        # Also, if it makes sense, I would rather call this self.integrand_list
        self.fun_list = [self]

    # Magic Methods. Makes self[i]==self.fun_list[i]
    def __len__(self):
        return len(self.fun_list)

    def __iter__(self):
        for fun in self.fun_list:
            yield fun

    def __getitem__(self, i):
        return self.fun_list[i]

    def __setitem__(self, i, val):
        self.fun_list[i] = val

    def __repr__(self):
        return univ_repr(self, 'fun_list')

    # Because Python does not require pure virtual functions, I consider it easier to avoid them altogether
    # I will leave this up to you, you can revert this and use ABC if you want
    # There are circumstances where pure virtual classes are beneficial, but I would avoid them in general
    #
    # Also, putting tex into comments is not good -- it's just hard to read.  Refer to variables by their names
    # And some of these references are unintelligible in current form -- \fu, \vc ??
    def g(self, x, coords_in_sequence): # original function to be integrated
        """Elementwise evaluation of the integrand at

        :param x: nodes, $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix
        :type x: array with shape(n, self.dim)
        :param coords_in_sequence: set of those coordinates in sequence needed, $\fu$
        :type coords_in_sequence: array with shape(n, self.dim)
        :return: matrix with values $f(\vx_{\fu,i},\vc)$ where if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise
        :rtype: array with shape(n, p)

        """
        raise NotImplementedError

    def transform_variable(self, measure_list, distribution_list):
        """
        This method performs the necessary variable transformation to put the
        original function in the form required by the discrete distribution
        object starting from the original measure object

        :param measure_list: the measure defining the integral
        :type measure_list: sub-class of algorithms.distribution.Measure
        :param distribution_list: the discrete distribution object from which samples are drawn
        :type distribution_list: sub-class of algorithms.distribution.DiscreteDistribution
        :return: Probably should be none, but right now it's returning this object
        :rtype: IntegrandBase

        """

        # NOTE(Mike)
        # I'm not really sure what's going on here as far as the goal
        # Also, the use of lambda functions here seems like a bit heavy
        # Usually lambda is reserved for wrapping small things, not something as significant as this
        # May way to reconsider at some point
        #
        # Also, do these self[ii].f items have some required structure?  There are many different calling sequences
        #
        # Also, timeDiff argument name should be lowercase

        for k, (integrand, measure, distribution) in enumerate(zip(self.fun_list, measure_list, distribution_list)):
            integrand.dimension = distribution.trueD.dimension
            if measure.measureName == distribution.trueD.measureName:
                integrand.f = lambda x, ci, i=k: self.fun_list[i].g(x, ci)
            elif measure.measureName == 'iid_zmean_gaussian' and distribution.trueD.measureName == 'std_gaussian':
                v = measure.measureData['variance']
                integrand.f = lambda x, ci, var=v, i=k: self.fun_list[i].g(x * np.sqrt(var), ci)
            elif measure.measureName == 'brownian_motion' and distribution.trueD.measureName == 'std_gaussian':
                td = np.diff(np.insert(measure.measureData['timeVector'], 0, 0))

                def transformed_integrand(x, ci, timeDiff=td, i=k):
                    return self.fun_list[i].g(np.cumsum(x * np.sqrt(timeDiff), 1), ci)
                integrand.f = transformed_integrand
            elif measure.measureName == 'iid_zmean_gaussian' and distribution.trueD.measureName == 'std_uniform':
                v = measure.measureData['variance']
                integrand.f = lambda x, ci, var=v, i=k: self.fun_list[i].g(np.sqrt(var) * norm.ppf(x), ci)
            elif measure.measureName == 'brownian_motion' and distribution.trueD.measureName == 'std_gaussian':
                td = np.diff(np.insert(measure.measureData['timeVector'], 0, 0))

                def transformed_integrand(x, ci, timeDiff=td, i=k):
                    return self.fun_list[i].g(np.cumsum(norm.ppf(x) * np.sqrt(timeDiff), 1), ci)
                integrand.f = transformed_integrand
            else:
                raise ValueError(f'{measure.measureName}, {distribution.trueD.measureName} not available')

        # Left this in to confirm, but if the above is correct then this can be cut
        # for ii in range(len(self)):
        #     self[ii].dimension = distribution_list[ii].trueD.dimension # the function needs the dimension
        #     if measure_list[ii].measureName==distribution_list[ii].trueD.measureName:
        #         self[ii].f = lambda xu,coordIdex,i=ii: self[i].g(xu,coordIdex)
        #     elif measure_list[ii].measureName== 'iid_zmean_gaussian' and distribution_list[ii].trueD.measureName== 'std_gaussian': # multiply by the likelihood ratio
        #         this_var = measure_list[ii].measureData['variance']
        #         self[ii].f = lambda xu,coordIndex,var=this_var,i=ii: self[i].g(xu*sqrt(var),coordIndex)
        #     elif measure_list[ii].measureName== 'brownian_motion' and distribution_list[ii].trueD.measureName== 'std_gaussian':
        #         timeDiff = diff(insert(measure_list[ii].measureData['timeVector'], 0, 0))
        #         self[ii].f = lambda xu,coordIndex,timeDiff=timeDiff,i=ii: self[i].g(cumsum(xu*sqrt(timeDiff),1),coordIndex)
        #     elif measure_list[ii].measureName== 'iid_zmean_gaussian' and distribution_list[ii].trueD.measureName== 'std_uniform':
        #         this_var = measure_list[ii].measureData['variance']
        #         self[ii].f = lambda xu,coordIdex,var=this_var,i=ii: self[i].g(sqrt(var)*norm.ppf(xu),coordIdex)
        #     elif measure_list[ii].measureName== 'brownian_motion' and distribution_list[ii].trueD.measureName== 'std_uniform':
        #         timeDiff = diff(insert(measure_list[ii].measureData['timeVector'], 0, 0))
        #         self[ii].f = lambda xu,coordIndex,timeDiff=timeDiff,i=ii: self[i].g(cumsum(norm.ppf(xu)*sqrt(timeDiff),1),coordIndex)
        #     else:
        #         raise Exception("Variable transformation not performed")

        return self
    # I feel like this function should not need to return anything, since the change is being done in place ...
