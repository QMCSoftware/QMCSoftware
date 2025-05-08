from .abstract_integrand import AbstractIntegrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np
from scipy.stats import norm 


class EuropeanOption(AbstractIntegrand):
    r"""
    European financial option.
    
    - Start price $S_0$
    - Strike price $K$
    - Interest rate $r$ 
    - Volatility $\sigma$
    - Monitoring times $\boldsymbol{\tau} = (\tau_1,\dots,\tau_d)^T$ with $\tau_d$ the final (exercise) time and, for $d>1$,  $\tau_1=0$. 

    Define the [geometric brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) as 

    $$\boldsymbol{G}(\boldsymbol{t}) = S_0 e^{(r-\sigma^2/2)\boldsymbol{\tau}+\sigma\boldsymbol{t}}, \qquad \boldsymbol{T} \sim \mathcal{N}(\boldsymbol{0},\mathsf{\Sigma})$$

    where $\boldsymbol{T}$ is a standard Brownian motion so $\mathsf{\Sigma} = \left(\min\{\tau_j,\tau_{j'}\}\right)_{j,j'=1}^d$.

    Letting $G_d(\boldsymbol{t})$ denote the last component of $\boldsymbol{G}(\boldsymbol{t})$, the payoff of a *call* option is 

    $$P(\boldsymbol{t}) = \max\{G_d(\boldsymbol{t})-K,0\}$$

    and the payoff of a *put* option is 
    
    $$P(\boldsymbol{t}) = \max\{K-G_d(\boldsymbol{t}),0\}.$$

    The discounted payoff is 

    $$g(\boldsymbol{t}) = P(\boldsymbol{t})e^{-r \tau_d}.$$

    Examples:
        >>> integrand = EuropeanOption(DigitalNetB2(4,seed=7))
        >>> x = integrand.discrete_distrib.gen_samples(2**10)
        >>> y = integrand.f(x)
        >>> print("%.4f"%y.mean())
        4.2327
        >>> print("%.4f"%integrand.get_exact_value())
        4.2115
        >>> integrand
        EuropeanOption (AbstractIntegrand)
            volatility      2^(-1)
            call_put        call
            start_price     30
            strike_price    35
            interest_rate   0
        >>> integrand.true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.25 0.5  0.75 1.  ]
            drift           0
            mean            [0. 0. 0. 0.]
            covariance      [[0.25 0.25 0.25 0.25]
                            [0.25 0.5  0.5  0.5 ]
                            [0.25 0.5  0.75 0.75]
                            [0.25 0.5  0.75 1.  ]]
            decomp_type     PCA

        With independent replications

        >>> integrand = EuropeanOption(DigitalNetB2(4,seed=7,replications=2**4))
        >>> x = integrand.discrete_distrib.gen_samples(2**6)
        >>> x.shape
        (16, 64, 4)
        >>> y = integrand.f(x)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1) 
        >>> muhats.shape 
        (16,)
        >>> print("%.4f"%muhats.mean())
        4.2306
    """
                          
    def __init__(self, sampler, volatility=0.5, start_price=30, strike_price=35, interest_rate=0, t_final=1, call_put='call'):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            volatility (float): $\sigma$.
            start_price (float): $S_0$.
            strike_price (float): $K$.
            interest_rate (float): $r$.
            t_final (float): $\tau_d$.
            call_put (str): Either `'call'` or `'put'`. 
        """
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.t_final = t_final
        self.sampler = sampler
        self.true_measure = BrownianMotion(self.sampler,t_final=self.t_final)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.discount_factor = np.exp(-self.interest_rate*self.t_final)
        super(EuropeanOption,self).__init__(dimension_indv=(),dimension_comb=(),parallel=False)  

    def g(self, t):
        gbm = self.start_price * np.exp((self.interest_rate-self.volatility**2/2)*self.true_measure.time_vec+self.volatility*t)
        gbm = gbm*(gbm>0).cumprod(-1) # if a path hits 0, set remaining values in the path to 0
        if self.call_put == 'call':
            payoff = np.maximum(gbm[...,-1]-self.strike_price,0)
        else: # put
            payoff = np.maximum(self.strike_price-gbm[...,-1],0)
        discounted_payoff = payoff*self.discount_factor
        return discounted_payoff
    
    def get_exact_value(self):
        """
        Compute the exact analytic value of the European call/put option fair price. 

        Returns: 
            mean (float): Exact value of the integral. 
        """
        denom = self.volatility*np.sqrt(self.t_final)
        decay = self.strike_price*self.discount_factor
        if self.call_put == 'call':
            term1 = np.log(self.start_price/self.strike_price)+(self.interest_rate+self.volatility**2/2)*self.t_final
            term2 = np.log(self.start_price/self.strike_price)+(self.interest_rate-self.volatility**2/2)*self.t_final
            fp = self.start_price * norm.cdf(term1/denom)-decay*norm.cdf(term2/denom)
        elif self.call_put == 'put':
            term1 = np.log(self.strike_price/self.start_price)-(self.interest_rate-self.volatility**2/2)*self.t_final
            term2 = np.log(self.strike_price/self.start_price)-(self.interest_rate+self.volatility**2/2)*self.t_final
            fp = decay*norm.cdf(term1/denom)-self.start_price*norm.cdf(term2/denom)
        return fp
    
    def _spawn(self, level, sampler):
        return EuropeanOption(
            sampler=sampler,
            volatility=self.volatility,
            start_price=self.start_price,
            strike_price=self.strike_price,
            interest_rate=self.interest_rate,
            t_final=self.t_final,
            call_put=self.call_put)
