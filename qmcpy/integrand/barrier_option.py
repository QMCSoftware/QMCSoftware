from qmcpy import*
from numpy import*
from ._integrand import Integrand
from ..util import ParameterError

class BarrierOption(Integrand):

    def __init__(self, n = 4096, volatility=0.2, start_price=30, strike_price=35, barrier_price = 38,
        interest_rate=0.05, t_final=1, observations = 52, call_put='call',in_out = 'in'):

        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'barrier_price', 'interest_rate','in_out']
        self.volatility = volatility
        self.strike_price = strike_price
        if(barrier_price == start_price):
            raise ParameterError("Barrier Price must be greater or less than the start price. They can't be equal.")
        self.barrier_price = barrier_price
        self.start_price = start_price
        self.interest_rate = interest_rate
        self.t_final = t_final
        self.observations = observations
        if(call_put.lower() == 'call' or call_put.lower() == 'put'):
            self.call_put = call_put.lower()
        else:
            raise ParameterError("call_put must be 'call' or 'put'")
        if(in_out.lower() == 'in' or in_out.lower() == 'out'):
            self.in_out = in_out.lower()
        else:
            raise ParameterError("in_out must be 'in' or 'out'")
        if(self.start_price < self.barrier_price):
            self.up = True
        else:
            self.up = False
        if(self.in_out == 'in'):
            self.i = True
        else:
            self.i = False
        if(self.call_put == 'call'):
            self.call = True
        else:
            self.call = False
        self.sampler = Sobol(self.observations)
        self.n = n
        self.true_measure = BrownianMotion(self.sampler,self.t_final)
        self.t = linspace(1/self.observations,self.t_final, self.observations)
        super(BarrierOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)   

    def get_discounted_payoff(self):
        stock_values = self.start_price*exp((self.interest_rate-0.5*self.volatility**2)*self.t+self.volatility*self.true_measure.gen_samples(self.n))
        print("Stock Values\n", stock_values)
        expected_stock = stock_values[:,self.observations - 1]
        print("Expected Stocks\n", expected_stock)
        if(self.call):
            disc_payoff = expected_stock - self.strike_price
        else:
            disc_payoff = self.strike_price - expected_stock
        print("Payoff without activation\n",disc_payoff)
        if(self.up and self.i):
            bar_flag = stock_values >= self.barrier_price
            print("Pre-barrier values\n",bar_flag)
            bar_flag = bar_flag.sum(axis = 1) > 0
        elif(self.up and (self.i is False)):
            bar_flag = stock_values < self.barrier_price
            print("Pre-barrier values\n",bar_flag)
            bar_flag = bar_flag.sum(axis = 1) == self.observations
        elif((self.up is False) and self.i):
            bar_flag = stock_values <= self.barrier_price
            print("Pre-barrier values\n",bar_flag)
            bar_flag = bar_flag.sum(axis = 1) > 0
        else:
            bar_flag = stock_values > self.barrier_price
            print("Pre-barrier values\n",bar_flag)
            bar_flag = bar_flag.sum(axis = 1) == self.observations
        print("Activating/Deactivation array\n",bar_flag)
        disc_payoff = disc_payoff*bar_flag
        print("Discounted payoff after activation\n", disc_payoff)
        disc_payoff = maximum(zeros(disc_payoff.size), disc_payoff)
        print("Final Discounted Payoff vector\n", disc_payoff)
        return disc_payoff.mean()
            
