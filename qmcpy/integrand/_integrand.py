from ..util import MethodImplementationError, _univ_repr, ParameterError
from ..true_measure._true_measure import TrueMeasure
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from numpy import *
import os
import multiprocessing
from itertools import repeat


class Integrand(object):
    """ Integrand abstract class. DO NOT INSTANTIATE. """

    def __init__(self, dprime, parallel):
        """
        Args:
            dprime (tuple): function output dimension shape.
            parallel (int): If parallel is False, 0, or 1: function evaluation is done in serial fashion.
                Otherwise, parallel specifies the number of CPUs used by multiprocessing.Pool.
                Passing parallel=True sets the number of CPUs equal to os.cpu_count().
        """
        prefix = 'A concrete implementation of Integrand must have '
        self.d = self.true_measure.d
        self.dprime = (dprime,) if isinstance(dprime,int) else tuple(dprime)
        cpus = os.cpu_count()
        self.parallel = cpus if parallel is True else int(parallel)
        self.parallel = 0 if self.parallel==1 else self.parallel
        if self.parallel>cpus:
            raise ParameterError("parallel must be less than %d, the number of CPUs on this machine."%cpus)
        if not (hasattr(self, 'sampler') and isinstance(self.sampler,(TrueMeasure,DiscreteDistribution))):
            raise ParameterError(prefix + 'self.sampler, a TrueMeasure or DiscreteDistributioninstance')
        if not (hasattr(self, 'true_measure') and isinstance(self.true_measure,TrueMeasure)):
            raise ParameterError(prefix + 'self.true_measure, a TrueMeasure instance')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'leveltype'):
            self.leveltype = 'single'
        if not hasattr(self,'max_level'):
            self.max_level = inf
        if not hasattr(self,'discrete_distrib'):
            self.discrete_distrib = self.true_measure.discrete_distrib
        if self.true_measure.transform!=self.true_measure and \
           not (self.true_measure.range==self.true_measure.transform.range).all():
            raise ParameterError("The range of the composed transform is not compatibe with this true measure")
        self.EPS = finfo(float).eps

    def g(self, t, *args, **kwargs):
        """
        ABSTRACT METHOD for original integrand to be integrated.

        Args:
            t (ndarray): n x d array of samples to be input into original integrand.

        Return:
            ndarray: n vector of function evaluations
        """
        raise MethodImplementationError(self, 'g')

    def f(self, x, periodization_transform='NONE', compute_flags=None, *args, **kwargs):
        """
        Evaluate transformed integrand based on true measures and discrete distribution

        Args:
            x (ndarray): n x d array of samples from a discrete distribution
            periodization_transform (str): periodization transform
            compute_flags (ndarray): TODO
            *args: other ordered args to g
            **kwargs (dict): other keyword args to g

        Return:
            ndarray: length n vector of function evaluations
        """
        periodization_transform = 'NONE' if periodization_transform is None else periodization_transform.upper()
        compute_flags = tile(1,self.dprime) if compute_flags is None else atleast_1d(compute_flags)
        n,d = x.shape
        # parameter checks
        if self.discrete_distrib.mimics != 'StdUniform' and periodization_transform!='NONE':
            raise ParameterError('''
                Applying a periodization transform currently requires a discrete distribution 
                that mimics a standard uniform measure.''')
        # periodization transform
        if periodization_transform == 'NONE':
            xp = x
            wp = ones(n,dtype=float)
        elif periodization_transform == 'BAKER': # Baker's transform
            xp = 1 - 2 * abs(x - 1 / 2)
            wp = ones(n,dtype=float)
        elif periodization_transform == 'C0': # C^0 transform
            xp = 3 * x ** 2 - 2 * x ** 3
            wp = prod(6 * x * (1 - x), 1)
        elif periodization_transform == 'C1': # C^1 transform
            xp = x ** 3 * (10 - 15 * x + 6 * x ** 2)
            wp = prod(30 * x ** 2 * (1 - x) ** 2, 1)
        elif periodization_transform == 'C1SIN': # Sidi C^1 transform
            xp = x - sin(2 * pi * x) / (2 * pi)
            wp = prod(2 * sin(pi * x) ** 2, 1)
        elif periodization_transform == 'C2SIN': # Sidi C^2 transform
            xp = (8 - 9 * cos(pi * x) + cos(3 * pi * x)) / 16 # psi3
            wp = prod( (9 * sin(pi * x) * pi - sin(3 * pi * x) * 3 * pi) / 16 , 1) # psi3_1
        elif periodization_transform == 'C3SIN': # Sidi C^3 transform
            xp = (12 * pi * x - 8 * sin(2 * pi * x) + sin(4 * pi * x)) / (12 * pi) # psi4
            wp = prod( (12 * pi - 8 * cos(2 * pi * x) * 2 * pi + sin(4 * pi * x) * 4 * pi) / (12 * pi), 1) # psi4_1
        else:
            raise ParameterError("The %s periodization transform is not implemented"%periodization_transform)
        wp = wp.reshape(n)
        if periodization_transform in ['C1','C1SIN','C2SIN','C3SIN']:
            xp[xp<=0] = self.EPS
            xp[xp>=1] = 1-self.EPS
        # function evaluation with chain rule
        y = empty((n,)+self.dprime,dtype=float)
        if self.true_measure == self.true_measure.transform:
            # jacobian*weight/pdf will cancel so f(x) = g(\Psi(x))
            xtf = self.true_measure._transform(xp) # get transformed samples, equivalent to self.true_measure._transform_r(x)
            y[:] = self._g(xtf,compute_flags,*args,**kwargs)
        else: # using importance sampling --> need to compute pdf, jacobian(s), and weight explicitly
            pdf = self.discrete_distrib.pdf(xp).reshape(n) # pdf of samples
            xtf,jacobians = self.true_measure.transform._jacobian_transform_r(xp) # compute recursive transform+jacobian
            jacobians = jacobians.reshape(n)
            weight = self.true_measure._weight(xtf).reshape(n) # weight based on the true measure
            gvals = self._g(xtf,compute_flags,*args,**kwargs)
            for i in range(n): y[i] = gvals[i]*weight[i]/pdf[i]*jacobians[i]
        # account for periodization weight
        for i in range(n): y[i] = y[i]*wp[i]
        return y

    def _g(self,t,compute_flags,*args,**kwargs):
        n = len(t)
        kwargs['compute_flags'] = compute_flags
        if self.parallel:
            pool = multiprocessing.Pool(processes=self.parallel)
            y = pool.starmap(self._g2,zip(t,repeat((args,kwargs))))
            y = concatenate(y,dtype=float)
        else:
            y = self._g2(t,comb_args=(args,kwargs))
        y = y.reshape((n,)+self.dprime)
        return y

    def _g2(self,t,comb_args=((),{})):
        args = comb_args[0]
        kwargs = comb_args[1]
        t = atleast_2d(t)
        if self.dprime==(1,):
            kwargs = dict(kwargs)
            del kwargs['compute_flags']
        y = self.g(t,*args,**kwargs)
        return y

    def bound_fun(self, bound_low, bound_high):
        """
        Compute the bounds on the combined function based on bounds for the
        individual functions. Defaults to the identity where we essentially
        do not combine integrands, but instead integrate each function
        individually.

        Args:
            bound_low (ndarray): length Integrand.dprime lower error bound
            bound_high (ndarray): length Integrand.dprime upper error bound

        Return:
            (tuple) containing

            - (ndarray): lower bound on function combining estimates
            - (ndarray): upper bound on function combining estimates
            - (ndarray): bool flags to override sufficient combined integrand estimation, e.g., when approximating a ratio of integrals, if the denominator's bounds straddle 0, then returning True here forces ratio to be flagged as insufficiently approximated.
        """
        return bound_low, bound_high, array([False])

    def dependency(self, flags_comb):
        """
        takes a vector of indicators of weather of not
        the error bound is satisfied for combined integrands and which returns flags for individual integrands.
        For example, if we are taking the ratio of 2 individual integrands, then getting flag_comb=True means the ratio
        has not been approximated to within the tolerance, so the dependency function should return [True,True]
        indicating that both the numerator and denominator integrands need to be better approximated.
        Args:
            flags_comb (bool ndarray): flags indicating weather the combined integrals are insufficiently approximated

        Return:
            (bool ndarray): length (Integrand.dprime) flags for individual integrands"""
        return flags_comb

    def _dimension_at_level(self, level):
        """
        ABSTRACT METHOD to return the dimension of samples to generate at level l.
        This method only needs to be implemented for multi-level integrands where
        the dimension changes depending on the level.

        Will default to return the current dimension (not using multilevel methods).
        Overwrite this method for multilevel integrands

        Args:
            level (int): level at which to return the dimension

        Return:
            int: dimension at input level
        """
        if self.leveltype!='single':
            raise MethodImplementationError(self, '_dimension_at_level')
        return self.d

    def spawn(self, levels):
        """
        Spawn new instances of the current integrand at the specified levels.

        Args:
            levels (ndarray): array of levels at which to spawn new integrands

        Return:
            list: list of Integrands linked to newly spawned TrueMeasures and DiscreteDistributions
        """
        levels = array([levels]) if isscalar(levels) else array(levels)
        if (levels>self.max_level).any():
            raise ParameterError("requested spawn level exceeds max level")
        n_levels = len(levels)
        spawns = [None]*n_levels
        for l in range(n_levels):
            level = levels[l]
            new_dim = self._dimension_at_level(level)
            tm_spawn = self.sampler.spawn(s=1,dimensions=[new_dim])[0]
            spawns[l] = self._spawn(level,tm_spawn)
        return spawns

    def _spawn(self, level, tm_spawn):
        """
        ABSTRACT METHOD, used by self.spawn

        Args:
            level (numpy.random.SeedSequence): level at which to spawn new instance
            tm_spawn (TrueMeasure): true measure spawn to use as new sampler

        Return:
            Integrand: spawn at this level
        """
        raise MethodImplementationError(self, '_spawn')

    def __repr__(self):
        return _univ_repr(self, "Integrand", self.parameters)
