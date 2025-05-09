from .abstract_integrand import AbstractIntegrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np
from scipy.stats import norm 


class FinancialOption(AbstractIntegrand):
    r"""
    Financial options.
    
    - Start price $S_0$
    - Strike price $K$
    - Interest rate $r$ 
    - Volatility $\sigma$
    - Equidistant monitoring times $\boldsymbol{\tau} = (\tau_1,\dots,\tau_d)^T$ with $\tau_d$ the final (exercise) time and $\tau_j = \tau_d j/d$. 

    Define the [geometric brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) as 

    $$\boldsymbol{S}(\boldsymbol{t}) = S_0 e^{(r-\sigma^2/2)\boldsymbol{\tau}+\sigma\boldsymbol{t}}, \qquad \boldsymbol{T} \sim \mathcal{N}(\boldsymbol{0},\mathsf{\Sigma})$$

    where $\boldsymbol{T}$ is a standard Brownian motion so $\mathsf{\Sigma} = \left(\min\{\tau_j,\tau_{j'}\}\right)_{j,j'=1}^d$.

    The discounted payoff is 

    $$g(\boldsymbol{t}) = P(\boldsymbol{S}(\boldsymbol{t}))e^{-r \tau_d}$$
    
    where the payoff function $P$ will be defined depending on the option.
    
    Below we wil luse $S_{-1}$ to denote the final element of $\boldsymbol{S}$, the value of the path at exercise time. 
    
    # European Options
    
    *European Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \max\{S_{-1}-K,0\}, \qquad P(\boldsymbol{S}) = \max\{K-S_{-1},0\}.$$

    # Asian Options

    An asian option considers the average value of an asset path across time. We use the trapazoidal rule to approximate either the *arithmetic mean* by 

    $$A(\boldsymbol{S}) = \frac{1}{d}\left[\frac{1}{2} S_0 + \sum_{j=1}^{d-1} S_j + \frac{1}{2} S_{-1}\right]$$ 
    
    or the *geometric mean* by 

    $$A(\boldsymbol{S}) = \left[\sqrt{S_0} \prod_{j=1}^{d-1} S_j \sqrt{S_{-1}}\right]^{1/d}.$$

    *Asian Call and Put Option* have respective payoffs 

    $$P(\boldsymbol{S}) = \max\{A(\boldsymbol{S})-K,0\}, \qquad P(\boldsymbol{S}) = \max\{K-A(\boldsymbol{S}),0\}.$$

    # Barrier Options 

    - Barrier $B$. 
    
    *In* options are activate when the path crosses the barrier $B$, while *out* options are activated only if the path never crosses the barrier $B$.
    An *up* option satisfies $S_0<B$ while a *down* option satisfies $S_0>B$, both indicating the direction of the barrier from the start price. 

    *Barrier Up-In Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{any } \boldsymbol{S} \geq B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{any } \boldsymbol{S} \geq B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    *Barrier Up-Out Call and Put Options* have respective payoffs 

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{all } \boldsymbol{S} < B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{all } \boldsymbol{S} < B \\ 0, & \mathrm{otherwise} \end{cases}.$$
    
    *Barrier Down-In Call and Put Options* have respective payoffs 

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{any } \boldsymbol{S} \leq B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{any } \boldsymbol{S} \leq B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    *Barrier Down-Out Call and Put Options* have respective payoffs 

    $$P(\boldsymbol{S}) = \begin{cases} \max\{S_{-1})-K,0\}, & \text{all } \boldsymbol{S} > B \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) = \begin{cases} \max\{K-S_{-1}),0\}, & \text{all } \boldsymbol{S} > B \\ 0, & \mathrm{otherwise} \end{cases}.$$

    # Lookback Options 

    *Lookback Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = S_{-1}-\min(\boldsymbol{S}), \qquad P(\boldsymbol{S}) = \max(\boldsymbol{S})-S_{-1}.$$
    
    # Digital Option 

    - Payout $\rho$. 

    *Digital Call and Put Options* have respective payoffs

    $$P(\boldsymbol{S}) = \begin{cases} \rho, & S_{-1} \geq K \\ 0, & \mathrm{otherwise} \end{cases}, \qquad P(\boldsymbol{S}) =  \begin{cases} \rho, & S_{-1} \leq K \\ 0, & \mathrm{otherwise} \end{cases}.$$

    # Multilevel Options

    - Initial level $\ell_0 \geq 0$. 
    - Level $\ell \geq \ell_0$.
    
    Let $\boldsymbol{S}_\mathrm{fine}=\boldsymbol{S}$ be the *fine* full path. For $\ell>\ell_0$ write the *coarse* path as $\boldsymbol{S}_\mathrm{coarse} = (S_j)_{j \text{ even}}$ which only considers every other element of $\boldsymbol{S}$. 
    In this multilevel setting the payoff is

    $$P_\ell(\boldsymbol{S}) = \begin{cases} P(\boldsymbol{S}_\mathrm{fine}), & \ell = \ell_0, \\ P(\boldsymbol{S}_\mathrm{fine})-P(\boldsymbol{S}_\mathrm{coarse}), & \ell > \ell_0 \end{cases}.$$

    Cancellations from the telescoping sum allow us to write 

    $$\lim_{\ell \to \infty} P_\ell = P_{\ell_0} + \sum_{\ell=\ell_0+1}^\infty P_\ell.$$

    Examples:
        >>> integrand = FinancialOption(DigitalNetB2(dimension=3,seed=7),option="EUROPEAN")
        >>> x = integrand.discrete_distrib.gen_samples(2**10)
        >>> y = integrand.f(x)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean())
        4.1996
        >>> print("%.4f"%integrand.get_exact_value())
        4.2115
        >>> integrand
        FinancialOption (AbstractIntegrand)
            option          EUROPEAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
        >>> integrand.true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.333 0.667 1.   ]
            drift           0
            mean            [0. 0. 0.]
            covariance      [[0.333 0.333 0.333]
                            [0.333 0.667 0.667]
                            [0.333 0.667 1.   ]]
            decomp_type     PCA

        >>> integrand = FinancialOption(DigitalNetB2(dimension=64,seed=7),option="ASIAN")
        >>> x = integrand.discrete_distrib.gen_samples(2**10)
        >>> y = integrand.f(x)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean())
        1.7996

        With independent replications

        >>> integrand = FinancialOption(DigitalNetB2(dimension=64,seed=7,replications=2**4),option="ASIAN")
        >>> x = integrand.discrete_distrib.gen_samples(2**6)
        >>> x.shape
        (16, 64, 64)
        >>> y = integrand.f(x)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1) 
        >>> muhats.shape 
        (16,)
        >>> print("%.4f"%muhats.mean())
        1.7923

        Multi-level options 

        >>> seed_seq = np.random.SeedSequence(7) 
        >>> initial_level = 3
        >>> num_levels = 4
        >>> ns = [2**11,2**10,2**9,2**8]
        >>> integrands = [FinancialOption(DigitalNetB2(dimension=2**l,seed=seed_seq.spawn(1)[0]),option="ASIAN",level=l,initial_level=initial_level) for l in range(initial_level,initial_level+num_levels)]
        >>> xs = [integrands[l].discrete_distrib.gen_samples(ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("xs[%d].shape = %s"%(l,xs[l].shape))
        xs[0].shape = (2048, 8)
        xs[1].shape = (1024, 16)
        xs[2].shape = (512, 32)
        xs[3].shape = (256, 64)
        >>> ys = [integrands[l].f(xs[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 2048)
        ys[1].shape = (2, 1024)
        ys[2].shape = (2, 512)
        ys[3].shape = (2, 256)
        >>> ymeans = np.stack([(ys[l][0]-ys[l][1]).mean(-1) for l in range(num_levels)])
        >>> ymeans
        array([ 1.77827205e+00,  3.58130113e-03,  2.51227835e-03, -8.34836713e-04])
        >>> print("%.4f"%ymeans.sum())
        1.7835

        Multi-level options with independent replications
         
        >>> seed_seq = np.random.SeedSequence(7) 
        >>> initial_level = 3
        >>> num_levels = 4
        >>> ns = [2**7,2**6,2**5,2**4]
        >>> integrands = [FinancialOption(DigitalNetB2(dimension=2**l,seed=seed_seq.spawn(1)[0],replications=2**4),option="ASIAN",level=l,initial_level=initial_level) for l in range(initial_level,initial_level+num_levels)]
        >>> xs = [integrands[l].discrete_distrib.gen_samples(ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("xs[%d].shape = %s"%(l,xs[l].shape))
        xs[0].shape = (16, 128, 8)
        xs[1].shape = (16, 64, 16)
        xs[2].shape = (16, 32, 32)
        xs[3].shape = (16, 16, 64)
        >>> ys = [integrands[l].f(xs[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 16, 128)
        ys[1].shape = (2, 16, 64)
        ys[2].shape = (2, 16, 32)
        ys[3].shape = (2, 16, 16)
        >>> muhats = np.stack([(ys[l][0]-ys[l][1]).mean(-1) for l in range(num_levels)])
        >>> muhats.shape
        (4, 16)
        >>> muhathat = muhats.mean(-1)
        >>> muhathat
        array([ 1.78967878,  0.02157042,  0.0024001 , -0.00278741])
        >>> print("%.4f"%muhathat.sum())
        1.8109

    **References:**

    1.  M.B. Giles.  
        Improved multilevel Monte Carlo convergence using the Milstein scheme.  
        343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008.  
        [http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf](http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf).
    """
                          
    def __init__(self, 
                 sampler,
                 option = "ASIAN", 
                 call_put = 'CALL',
                 volatility = 0.5,
                 start_price = 30,
                 strike_price = 35,
                 interest_rate = 0,
                 t_final = 1,
                 decomp_type = 'PCA',
                 level = None,
                 initial_level = 0,
                 asian_mean = "ARITHMETIC",
                 barrier_in_out = "IN",
                 barrier_price = 38,
                 digital_payout = 10):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            option (str): Option type in `['ASIAN', 'EUROPEAN', 'BARRIER', 'LOOKBACK', 'DIGITAL']`
            call_put (str): Either `'CALL'` or `'PUT'`. 
            volatility (float): $\sigma$.
            start_price (float): $S_0$.
            strike_price (float): $K$.
            interest_rate (float): $r$.
            t_final (float): $\tau_d$.
            decomp_type (str): Method for decomposition for covariance matrix. Options include
             
                - `'PCA'` for principal component analysis, or 
                - `'Cholesky'` for cholesky decomposition.
            level (Union[None,int]): Level for multilevel problems 
            initial_level (Union[None,int]): First level for multilevel problems.
            asian_mean (str): Either `'ARITHMETIC'` or `'GEOMETRIC'`.
            barrier_in_out (str): Either `'IN'` or `'OUT'`. 
            barrier_price (float): $B$. 
            digital_payout (float): $\rho$. 
        """
        self.parameters = ['option', 'call_put', 'volatility', 'start_price', 'strike_price', 'interest_rate', 't_final']
        self.t_final = t_final
        self.sampler = sampler
        self.decomp_type = decomp_type
        self.true_measure = BrownianMotion(self.sampler,t_final=self.t_final,decomp_type=self.decomp_type)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.discount_factor = np.exp(-self.interest_rate*self.t_final)
        self.level = level
        self.initial_level = initial_level
        if self.level is not None: 
            self.parameters += ['level','initial_level']
            assert np.isscalar(self.level) and self.level%1==0 
            assert np.isscalar(self.initial_level) and self.initial_level%1==0
            self.level = int(self.level)
            self.initial_level = int(self.initial_level) 
            assert self.sampler.d==2**self.level, "the dimension of the sampler must equal 2^level = %d"%(2**self.level)
            assert 0<=self.initial_level<=self.level, "require 0 <= initial_level (%d) <= level (%d)."%(self.initial_level,self.level)
        self.call_put = str(call_put).upper()
        assert self.call_put in ['CALL','PUT'], "invalid call_put = %s"%self.call_put
        self.option = str(option).upper()
        if self.option=="EUROPEAN":
            self.payoff = self.payoff_european_call if self.call_put=='CALL' else self.payoff_european_put
        elif self.option=="ASIAN":
            self.parameters += ['asian_mean']
            self.asian_mean = str(asian_mean).upper()
            if self.asian_mean=="ARITHMETIC":
                self.payoff = self.payoff_asian_arithmetic_call if self.call_put=='CALL' else self.payoff_asian_arithmetic_put
            elif self.asian_mean=="GEOMETRIC":
                self.payoff = self.payoff_asian_geometric_call if self.call_put=='CALL' else self.payoff_asian_geometric_put
            else:
                raise ParameterError("invalid asian_mean = %s"%self.asian_mean)
        elif self.option=="BARRIER":
            self.parameters += ['barrier_in_out','barrier_up_down','barrier_price']
            assert np.isscalar(barrier_price)
            self.barrier_price = float(barrier_price)
            self.barrier_in_out = str(barrier_in_out).upper()
            self.barrier_up_down = "UP" if self.start_price<self.barrier_price else "DOWN"
            if self.barrier_in_out=="IN":
                if self.barrier_up_down=="UP":
                    self.payoff = self.payoff_barrier_in_up_call if self.call_put=='CALL' else self.payoff_barrier_in_up_put
                else:
                    self.payoff = self.payoff_barrier_in_down_call if self.call_put=='CALL' else self.payoff_barrier_in_down_put
            elif self.barrier_in_out=="OUT":
                if self.barrier_up_down=="UP":
                    self.payoff = self.payoff_barrier_out_up_call if self.call_put=='CALL' else self.payoff_barrier_out_up_put
                else:
                    self.payoff = self.payoff_barrier_out_down_call if self.call_put=='CALL' else self.payoff_barrier_out_down_put
            else:
                raise ParameterError("invalid barrier_in_out = %s"%self.barrier_in_out)
        elif self.option=="LOOKBACK":
            self.payoff = self.payoff_lookback_call if self.call_put=='CALL' else self.payoff_lookback_put
        elif self.option=="DIGITAL":
            assert np.isscalar(digital_payout) and digital_payout>0
            self.digital_payout = float(digital_payout)
            self.payoff = self.payoff_digital_call if self.call_put=='CALL' else self.payoff_digital_put
        else:
            raise ParameterError("invalid option type %s"%self.option)
        dim_shape = (2,) if self.level is not None else ()
        super(FinancialOption,self).__init__(dimension_indv=dim_shape,dimension_comb=dim_shape,parallel=False)  
    
    def g(self, t, **kwargs):
        gbm = self.gbm(t)
        discounted_payoffs = self.payoff(gbm)*self.discount_factor
        if self.level is not None:
            if self.level==self.initial_level:
                discounted_payoffs_coarse = np.zeros_like(discounted_payoffs)
            else: 
                discounted_payoffs_coarse = self.payoff(gbm[...,1::2])*self.discount_factor
            discounted_payoffs = np.stack([discounted_payoffs,discounted_payoffs_coarse])
        return discounted_payoffs
    
    def gbm(self, t):
        gbm = self.start_price*np.exp((self.interest_rate-self.volatility**2/2)*self.true_measure.time_vec+self.volatility*t)
        gbm *= (gbm>0).cumprod(-1) # if a path hits 0, set remaining values in the path to 0
        return gbm
    
    def payoff_european_call(self, gbm):
        return np.maximum(gbm[...,-1]-self.strike_price,0)
    
    def payoff_european_put(self, gbm):
        return np.maximum(self.strike_price-gbm[...,-1],0)
    
    def payoff_asian_arithmetic_call(self, gbm):
        return np.maximum((self.start_price/2+gbm[...,:-1].sum(-1)+gbm[...,-1]/2)/gbm.shape[-1]-self.strike_price,0)
    
    def payoff_asian_arithmetic_put(self, gbm):
        return np.maximum((self.strike_price-self.start_price/2+gbm[...,:-1].sum(-1)+gbm[...,-1]/2)/gbm.shape[-1],0)
    
    def payoff_asian_geometric_call(self, gbm):
        return np.maximum(np.exp((np.log(self.start_price)/2+np.log(gbm[...,:-1]).sum(-1)+np.log(gbm[...,-1])/2)/gbm.shape[-1])-self.strike_price,0)
    
    def payoff_asian_geometric_put(self, gbm):
        return np.maximum(self.strike_price-np.exp((np.log(self.start_price)/2+np.log(gbm[...,:-1]).sum(-1)+np.log(gbm[...,-1])/2)/gbm.shape[-1]),0)
    
    def payoff_barrier_in_up_call(self, gbm):
        v = gbm[...,-1]
        flag = (gbm>=self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag]-self.strike_price,0)
        return v
    
    def payoff_barrier_out_up_call(self, gbm):
        v = gbm[...,-1]
        flag = (gbm<self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag]-self.strike_price,0)
        return v
    
    def payoff_barrier_in_down_call(self, gbm):
        v = gbm[...,-1]
        flag = (gbm<=self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag]-self.strike_price,0)
        return v
    
    def payoff_barrier_out_down_call(self, gbm):
        v = gbm[...,-1]
        flag = (gbm>self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(v[flag]-self.strike_price,0)
        return v
    
    def payoff_barrier_in_up_put(self, gbm):
        v = gbm[...,-1]
        flag = (gbm>=self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price-v[flag],0)
        return v
    
    def payoff_barrier_out_up_put(self, gbm):
        v = gbm[...,-1]
        flag = (gbm<self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price-v[flag],0)
        return v
    
    def payoff_barrier_in_down_put(self, gbm):
        v = gbm[...,-1]
        flag = (gbm<=self.barrier_price).any(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price-v[flag],0)
        return v
    
    def payoff_barrier_out_down_put(self, gbm):
        v = gbm[...,-1]
        flag = (gbm>self.barrier_price).all(-1)
        v[~flag] = 0
        v[flag] = np.maximum(self.strike_price-v[flag],0)
        return v
    
    def payoff_lookback_call(self, gbm):
        return gbm[...,-1]-gbm.min(-1)
    
    def payoff_lookback_put(self, gbm):
        return gbm.max(-1)-gbm[...,-1]
    
    def payoff_digital_call(self, gbm):
        return np.where(gbm[...,-1]>=self.strike_price,self.digital_payout,0)
    
    def payoff_digital_put(self, gbm):
        return np.where(gbm[...,-1]<=self.strike_price,self.digital_payout,0)
    
    def get_exact_value(self):
        """
        Compute the exact analytic fair price of the option. Only supports `option='EUROPEAN'`.

        Returns: 
            mean (float): Exact value of the integral. 
        """
        assert self.option=="EUROPEAN", "exact value not supported for option = %s"%self.option
        denom = self.volatility*np.sqrt(self.t_final)
        decay = self.strike_price*self.discount_factor
        if self.call_put == 'CALL':
            term1 = np.log(self.start_price/self.strike_price)+(self.interest_rate+self.volatility**2/2)*self.t_final
            term2 = np.log(self.start_price/self.strike_price)+(self.interest_rate-self.volatility**2/2)*self.t_final
            fp = self.start_price*norm.cdf(term1/denom)-decay*norm.cdf(term2/denom)
        elif self.call_put == 'PUT':
            term1 = np.log(self.strike_price/self.start_price)-(self.interest_rate-self.volatility**2/2)*self.t_final
            term2 = np.log(self.strike_price/self.start_price)-(self.interest_rate+self.volatility**2/2)*self.t_final
            fp = decay*norm.cdf(term1/denom)-self.start_price*norm.cdf(term2/denom)
        else:
            raise ParameterError("Error parsing payoff self.call_put = %s"%self.call_put)
        return fp
    
    def _spawn(self, level, sampler):
        return FinancialOption(
            sampler=sampler,
            volatility=self.volatility,
            start_price=self.start_price,
            strike_price=self.strike_price,
            interest_rate=self.interest_rate,
            t_final=self.t_final,
            call_put=self.call_put,
            decomp_type=self.decomp_type,
            level=self.level, 
            initial_level=self.initial_level,
            asian_mean=self.asian_mean,
            barrier_in_out=self.barrier_in_out, 
            barrier_price=self.barrier_price,
            digital_payoff=self.digital_payoff)
