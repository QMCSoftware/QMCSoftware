from abc import ABC, abstractmethod

from numpy import cumsum, diff, insert, sqrt
from scipy.stats import norm

from .. import univ_repr


class Integrand(ABC):
    def __init__(self, nominal_value=None):
        """
        Specify and generate values :math:`f(\mathbf{x})` for :math:`\mathbf{x}
        \in \mathcal{X}`. Any sublcass of Integrand must include the method,
        g(self, x, coordIndex)

        Args:
            nominal_value: math:`c` such that :math:`(c, \ldots, c) \in \mathcal{X}`

        """
        super().__init__()
        self.nominalValue = nominal_value if nominal_value else 0
        self.f = None  # integrand handle of integrand after transformation
        self.dimension = 2  # dimension of the domain, :math:`d`
        self.fun_list = [self]

    # Abstract Methods
    @abstractmethod
    def g(self, x, coordIndex):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` row of an :math:`n \cdot |\mathfrak{u}|` matrix
            coordIndex: set of those coordinates in sequence needed, :math:`\mathfrak{u}`

        Returns:
            :math:`n \cdot p` matrix with values  :math:`f(\mathbf{x}_{\mathfrak{u},i},\mathbf{c})` where if :math:`\mathbf{x}_i' = (x_{i, \mathfrak{u}},\mathbf{c})_j`, then :math:`x'_{ij} = x_{ij}` for :math:`j \in \mathfrak{u}`, and :math:`x'_{ij} = c` otherwise

        """
        pass

    def transform_variable(self, msr_obj, dstr_obj):
        """
        This method performs the necessary variable transformation to put the
        original integrand in the form required by the DiscreteDistributon
        object starting from the original Measure object

        Args:
            msr_obj: the Measure object that defines the integral
            dstr_obj: the discrete distribution object that is sampled from

        Returns: transformed integrand

        """
        for i in range(len(self)):
            try: sample_from = dstr_obj[i].trueD.mimics # QuasiRandom sampling
            except: sample_from = type(dstr_obj[i].trueD).__name__ # IIDDistribution sampling
            transform_to = type(msr_obj[i]).__name__ # distribution the sampling attempts to mimic
            self[i].dimension = dstr_obj[i].trueD.dimension # the integrand needs the dimension
            if transform_to==sample_from: # no need to transform
                self[i].f = lambda xu,coordIdex,i=i: self[i].g(xu,coordIdex)
            elif transform_to== 'IIDZeroMeanGaussian' and sample_from== 'StdGaussian': # multiply by the likelihood ratio
                this_var = msr_obj[i].variance
                self[i].f = lambda xu,coordIndex,var=this_var,i=i: self[i].g(xu*sqrt(var),coordIndex)
            elif transform_to== 'IIDZeroMeanGaussian' and sample_from== 'StdUniform':
                this_var = msr_obj[i].variance
                self[i].f = lambda xu,coordIdex,var=this_var,i=i: self[i].g(sqrt(var)*norm.ppf(xu),coordIdex)
            elif transform_to== 'BrownianMotion' and sample_from== 'StdUniform':
                timeDiff = diff(insert(msr_obj[i].timeVector, 0, 0))
                self[i].f = lambda xu,coordIndex,timeDiff=timeDiff,i=i: self[i].g(cumsum(norm.ppf(xu)*sqrt(timeDiff),1),coordIndex)
            elif transform_to== 'BrownianMotion' and sample_from== 'StdGaussian':
                timeDiff = diff(insert(msr_obj[i].timeVector, 0, 0))
                self[i].f = lambda xu,coordIndex,timeDiff=timeDiff,i=i: self[i].g(cumsum(xu*sqrt(timeDiff),1),coordIndex)
            else:
                raise Exception("Variable transformation not performed")
        return self

    # Magic Methods. Makes self[i]==self.fun_list[i]
    def __len__(self): return len(self.fun_list)   
    def __iter__(self):
        for fun in self.fun_list: yield fun
    def __getitem__(self,i): return self.fun_list[i]
    def __setitem__(self,i,val): self.fun_list[i] = val
    def __repr__(self): return univ_repr(self,'fun_list')