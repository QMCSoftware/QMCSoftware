from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import Sobol
from ..util import ParameterError
from numpy import *
from scipy.stats import norm 


class EuropeanOption(Integrand):
    """
    European financial option. 

    >>> dd = Sobol(4,seed=7)
    >>> m = BrownianMotion(dd,drift=-1)
    >>> eo = EuropeanOption(m,call_put='put')
    >>> eo
    EuropeanOption (Integrand Object)
        volatility      2^(-1)
        call_put        put
        start_price     30
        strike_price    35
        interest_rate   0
    >>> x = dd.gen_samples(2**10)
    >>> y = eo.f(x)
    >>> y.mean()
    9.211371880941195
    """

    parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
                          
    def __init__(self, measure, volatility=0.5, start_price=30, strike_price=35,
        interest_rate=0, call_put='call'):
        """
        Args:
            measure (TrueMeasure): A BrownianMotion TrueMeasure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            call_put (str): 'call' or 'put' option
        """
        if not isinstance(measure,BrownianMotion):
            raise ParameterError('EuropeanCall measure must be a BrownianMotion instance')
        self.measure = measure
        self.distribution = self.measure.distribution
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.exercise_time = self.measure.time_vector[-1]
        super(EuropeanOption,self).__init__()        

    def g(self, x):
        """ See abstract method. """
        self.s = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.measure.time_vector + self.volatility * x)
        for xx,yy in zip(*where(self.s<0)): # if stock becomes <=0, 0 out rest of path
            self.s[xx,yy:] = 0
        if self.call_put == 'call':
            y_raw = maximum(self.s[:,-1] - self.strike_price, 0)
        else: # put
            y_raw = maximum(self.strike_price - self.s[:,-1], 0)
        y_adj = y_raw * exp(-self.interest_rate * self.exercise_time)
        return y_adj
    
    def get_exact_value(self):
        """
        Get the fair price of a European call/put option.
        
        Return:
            float: fair price
        """
        denom = self.volatility * sqrt(self.exercise_time)
        decay = self.strike_price * exp(-self.interest_rate * self.exercise_time)
        if self.call_put == 'call':
            term1 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate + self.volatility**2/2) * self.exercise_time
            term2 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate - self.volatility**2/2) * self.exercise_time
            fp = self.start_price * norm.cdf(term1/denom) - decay * norm.cdf(term2/denom)
        elif self.call_put == 'put':
            term1 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate - self.volatility**2/2) * self.exercise_time
            term2 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate + self.volatility**2/2) * self.exercise_time
            fp = decay * norm.cdf(term1/denom) - self.start_price * norm.cdf(term2/denom)
        return fp
    
    def plot(self, n=2**5, show=True, out=None):
        """
        Plot European Option price vs time.
        
        Args:
            n (int): self.gen_samples(n)
            show (bool): show the plot?
            out (str): file name to output image. If None, the image is not output
            
        Return:
            tuple: fig,ax from `fig,ax = pyplot.subplots()`
        """
        tvw0 = hstack((0,self.measure.time_vector)) # time vector including 0
        x = self.distribution.gen_samples(n)
        y = self.f(x)
        sw0 = hstack((self.start_price*ones((n,1)),self.s)) # x including 0 and time 0
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        fig,ax = pyplot.subplots()
        for i in range(n):
            ax.plot(tvw0,sw0[i])
        ax.axhline(y=self.strike_price, color='k', linestyle='--', label='Strike Price')
        ax.set_xlim([0,1])
        ax.set_xticks([0,1])
        ax.set_xlabel('Time')
        ax.set_ylabel('Option Price')
        ax.legend(loc='upper left')
        s = '$2^{%d}$'%log2(n) if log2(n)%1==0 else '%d'%n 
        ax.set_title(s+' Asset Price Paths')
        fig.tight_layout()
        if out: pyplot.savefig(out,dpi=250)
        if show: pyplot.show()
        return fig,ax
