from ..util import MethodImplementationError, _univ_repr, ParameterError
from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
import numpy as np
import os
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from itertools import repeat


class AbstractIntegrand(object):

    def __init__(self, dimension_indv, dimension_comb, parallel, threadpool=False):
        r"""
        Args:
            dimension_indv (tuple): Individual solution shape.
            dimension_comb (tuple): Combined solution shape. 
            parallel (int): Parallelization flag. 
                
                - When `parallel = 0` or `parallel = 1` then function evaluation is done in serial fashion.
                - `parallel > 1` specifies the number of processes used by `multiprocessing.Pool` or `multiprocessing.pool.ThreadPool`.
            
                Setting `parallel=True` is equivalent to `parallel = os.cpu_count()`.
            threadpool (bool): When `parallel > 1`: 
                
                - Setting `threadpool = True` will use `multiprocessing.pool.ThreadPool`.
                - Setting `threadpool = False` will use `setting multiprocessing.Pool`.
        """
        prefix = 'A concrete implementation of AbstractIntegrand must have '
        self.d = self.true_measure.d
        self.d_indv = (dimension_indv,) if isinstance(dimension_indv,int) else tuple(dimension_indv)
        self.d_comb = (dimension_comb,) if isinstance(dimension_comb,int) else tuple(dimension_comb)
        cpus = os.cpu_count()
        self.parallel = cpus if parallel is True else int(parallel)
        self.parallel = 0 if self.parallel==1 else self.parallel
        self.threadpool = threadpool
        if not (hasattr(self, 'sampler') and isinstance(self.sampler,(AbstractTrueMeasure,AbstractDiscreteDistribution))):
            raise ParameterError(prefix + 'self.sampler, a AbstractTrueMeasure or AbstractDiscreteDistribution instance')
        if not (hasattr(self, 'true_measure') and isinstance(self.true_measure,AbstractTrueMeasure)):
            raise ParameterError(prefix + 'self.true_measure, a AbstractTrueMeasure instance')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'leveltype'):
            self.leveltype = 'single'
        if not hasattr(self,'max_level'):
            self.max_level = np.inf
        if not hasattr(self,'discrete_distrib'):
            self.discrete_distrib = self.true_measure.discrete_distrib
        if self.true_measure.transform!=self.true_measure and \
           not (self.true_measure.range==self.true_measure.transform.range).all():
            raise ParameterError("The range of the composed transform is not compatible with this true measure")
        self.EPS = np.finfo(float).eps

    def g(self, t, compute_flags, *args, **kwargs):
        r"""
        *Abstract method* implementing the integrand as a function of the true measure.

        Args:
            t (np.ndarray): Inputs with shape `(*batch_shape, d)`.
            compute_flags (np.ndarray): Flags indicating which outputs require evaluation.  
                For example, if the vector function has 3 outputs and `compute_flags = [False, True, False]`, 
                then the function is only required to evaluate the second output and may leave the remaining outputs as `np.nan` values,  
                i.e., the outputs corresponding to `compute_flags` which are `False` will not be used in the computation.

        Returns:
            y (np.ndarray): function evaluations with shape `(*batch_shape, *dimension_indv)` where `dimension_indv` is the shape of the function outputs. 
        """
        raise MethodImplementationError(self, 'g')

    def f(self, x, periodization_transform='NONE', compute_flags=None, *args, **kwargs):
        r"""
        Function to evaluate the transformed integrand as a function of the discrete distribution.  
        Automatically applies the transformation determined by the true measure. 

        Args:
            x (np.ndarray): Inputs with shape `(*batch_shape, d)`.
            periodization_transform (str): Periodization transform. Options are: 

                - `False`: No periodizing transform, $\psi(x) = x$. 
                - `'BAKER'`: Baker tansform $\psi(x) = 1-2\lvert x-1/2 \rvert$.
                - `'C0'`: $C^0$ transform $\psi(x) = 3x^2-2x^3$.
                - `'C1'`: $C^1$ transform $\psi(x) = x^3(10-15x+6x^2)$.
                - `'C1SIN'`: Sidi $C^1$ transform $\psi(x) = x-\sin(2 \pi x)/(2 \pi)$. 
                - `'C2SIN'`: Sidi $C^2$ transform $\psi(x) = (8-9 \cos(\pi x)+\cos(3 \pi x))/16$.
                - `'C3SIN'`: Sidi $C^3$ transform $\psi(x) = (12\pi x-8\sin(2 \pi x) + \sin(4 \pi x))/(12 \pi)$.
            compute_flags (np.ndarray): Flags indicating which outputs require evaluation.  
                For example, if the vector function has 3 outputs and `compute_flags = [False, True, False]`, 
                then the function is only required to evaluate the second output and may leave the remaining outputs as `np.nan` values,  
                i.e., the outputs corresponding to `compute_flags` which are `False` will not be used in the computation.
            *args (tuple): Other ordered args to `g`.
            **kwargs (dict): Other keyword args to `g`.

        Returns:
            y (np.ndarray): function evaluations with shape `(*batch_shape, *dimension_indv)` where `dimension_indv` is the shape of the function outputs. 
        """
        periodization_transform = str(periodization_transform).upper()
        compute_flags = np.tile(1,self.d_indv) if compute_flags is None else np.atleast_1d(compute_flags)
        batch_shape = tuple(x.shape[:-1])
        d_indv_ndim = len(self.d_indv)
        if periodization_transform=="NONE": periodization_transform = "FALSE"
        if self.discrete_distrib.mimics != 'StdUniform' and periodization_transform!='NONE':
            raise ParameterError('''
                Applying a periodization transform currently requires a discrete distribution 
                that mimics a standard uniform measure.''')
        if periodization_transform == 'FALSE':
            xp = x
            wp = np.ones(batch_shape,dtype=float)
        elif periodization_transform == 'BAKER':
            xp = 1-2*abs(x-1/2)
            wp = np.ones(batch_shape,dtype=float)
        elif periodization_transform == 'C0':
            xp = 3*x**2-2*x**3
            wp = np.prod(6*x*(1-x),-1)
        elif periodization_transform == 'C1':
            xp = x**3*(10-15*x+6*x**2)
            wp = np.prod(30*x**2*(1-x)**2,-1)
        elif periodization_transform == 'C1SIN':
            xp = x - np.sin(2*np.pi*x)/(2*np.pi)
            wp = np.prod(2*np.sin(np.pi*x)**2,-1)
        elif periodization_transform == 'C2SIN':
            xp = (8-9*np.cos(np.pi*x)+np.cos(3*np.pi*x))/16
            wp = np.prod((9*np.sin(np.pi*x)*np.pi-np.sin(3*np.pi*x)*3*np.pi)/16,-1)
        elif periodization_transform=='C3SIN':
            xp = (12*np.pi*x-8*np.sin(2*np.pi*x)+np.sin(4*np.pi*x))/(12*np.pi)
            wp = np.prod((12*np.pi-8*np.cos(2*np.pi*x)*2*np.pi+np.sin(4*np.pi*x)*4*np.pi)/(12*np.pi),-1)
        else:
            raise ParameterError("The %s periodization transform is not implemented"%periodization_transform)
        if periodization_transform in ['C1','C1SIN','C2SIN','C3SIN']:
            xp[xp<=0] = self.EPS
            xp[xp>=1] = 1-self.EPS
        assert wp.shape==batch_shape
        assert xp.shape==x.shape
        # function evaluation with chain rule
        i = (...,)+(None,)*d_indv_ndim
        if self.true_measure==self.true_measure.transform:
            # jacobian*weight/pdf will cancel so f(x) = g(\Psi(x))
            xtf = self.true_measure._jacobian_transform_r(xp,return_weights=False) # get transformed samples, equivalent to self.true_measure._transform_r(x)
            assert xtf.shape==xp.shape
            y = self._g(xtf,compute_flags,*args,**kwargs)
        else: # using importance sampling --> need to compute pdf, jacobian(s), and weight explicitly
            pdf = self.discrete_distrib.pdf(xp) # pdf of samples
            assert pdf.shape==batch_shape
            xtf,jacobians = self.true_measure.transform._jacobian_transform_r(xp,return_weights=True) # compute recursive transform+jacobian
            assert xtf.shape==xp.shape 
            assert jacobians.shape==batch_shape
            weight = self.true_measure._weight(xtf) # weight based on the true measure
            assert weight.shape==batch_shape
            gvals = self._g(xtf,compute_flags,*args,**kwargs)
            assert gvals.shape==(batch_shape+self.d_indv)
            y = gvals*weight[i]/pdf[i]*jacobians[i]
        assert y.shape==(batch_shape+self.d_indv)
        # account for periodization weight
        y = y*wp[i]
        assert y.shape==(batch_shape+self.d_indv)
        return y

    def _g(self, t, compute_flags, *args, **kwargs):
        kwargs['compute_flags'] = compute_flags
        if self.parallel:
            pool = get_context(method='fork').Pool(processes=self.parallel) if self.threadpool==False else ThreadPool(processes=self.parallel) 
            y = pool.starmap(self._g2,zip(t,np.repeat((args,kwargs))))
            pool.close()
            y = np.concatenate(y,dtype=float)
        else:
            y = self._g2(t,comb_args=(args,kwargs))
        assert y.shape==(t.shape[:-1]+self.d_indv)
        return y

    def _g2(self, t, comb_args=((),{})):
        args = comb_args[0]
        kwargs = comb_args[1]
        if self.d_indv==():
            kwargs = dict(kwargs)
            del kwargs['compute_flags']
        y = self.g(t,*args,**kwargs)
        return y

    def bound_fun(self, bound_low, bound_high):
        """
        Compute the bounds on the combined function based on bounds for the
        individual functions.  

        Defaults to the identity where we essentially
        do not combine integrands, but instead integrate each function
        individually.

        Args:
            bound_low (np.ndarray): length AbstractIntegrand.d_indv lower error bound
            bound_high (np.ndarray): length AbstractIntegrand.d_indv upper error bound

        Returns:
            comb_bound_low (np.ndarray): Lower bound on function combining estimates.
            comb_bound_high (np.ndarray): Upper bound on function combining estimates.
            comb_compute_flags (np.ndarray): Bool flags to override sufficient combined integrand estimation,  
                e.g., when approximating a ratio of integrals, if the denominator's bounds straddle 0, then returning `True` here forces ratio to be flagged as insufficiently approximated.
        """
        if self.d_indv!=self.d_comb:
            raise ParameterError('''
                Set bound_fun explicitly. 
                The default bound_fun is the identity map. 
                Since the individual solution dimensions d_indv = %s does not equal the combined solution dimensions d_comb = %d, 
                QMCPy cannot infer a reasonable bound function.'''%(str(self.d_indv),str(self.d_comb)))
        return bound_low,bound_high

    def dependency(self, comb_flags):
        """
        Takes a vector of indicators of weather of not the error bound is satisfied for combined integrands and returns flags for individual integrands.  
        
        For example, if we are taking the ratio of 2 individual integrands, then getting flag_comb=True means the ratio
        has not been approximated to within the tolerance, so the dependency function should return [True,True]
        indicating that both the numerator and denominator integrands need to be better approximated.

        Args:
            comb_flags (np.ndarray): Bool flags indicating weather the combined integrals are insufficiently approximated.

        Returns:
            indv_flags (np.ndarray): Bool flags for individual integrands. 
        """
        return comb_flags if self.d_indv==self.d_comb else np.tile((comb_flags==False).any(),self.d_indv)

    def _dimension_at_level(self, level):
        """
        ABSTRACT METHOD to return the dimension of samples to generate at level l.
        This method only needs to be implemented for multi-level integrands where
        the dimension changes depending on the level.

        Will default to return the current dimension (not using multilevel methods).
        Overwrite this method for multilevel integrands

        Args:
            level (int): level at which to return the dimension

        Returns:
            int: dimension at input level
        """
        if self.leveltype!='single':
            raise MethodImplementationError(self, '_dimension_at_level')
        return self.d

    def spawn(self, levels):
        r"""
        Spawn new instances of the current integrand at the specified levels.

        Args:
            levels (np.ndarray): Vector of levels at which to spawn new integrands.

        Returns:
            integrand_spawns (list): AbstractIntegrand instances with newly spawned TrueMeasures and DiscreteDistributions
        """
        levels = np.array([levels]) if np.isscalar(levels) else np.array(levels)
        if (levels>self.max_level).any():
            raise ParameterError("requested spawn level exceeds max level")
        n_levels = len(levels)
        new_dims = np.array([self._dimension_at_level(level) for level in levels])
        tm_spawns = self.sampler.spawn(s=n_levels,dimensions=new_dims) 
        integrand_spawns = [None]*n_levels
        for l,level in enumerate(levels):
            integrand_spawns[l] = self._spawn(level,tm_spawns[l])
        return integrand_spawns

    def _spawn(self, level, tm_spawn):
        raise MethodImplementationError(self, '_spawn')

    def __repr__(self):
        return _univ_repr(self, "AbstractIntegrand", self.parameters)
