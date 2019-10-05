"""
Definition for class AsianCall, a subclass of Integrand
"""
from numpy import exp, maximum

from . import Integrand


class AsianCall(Integrand):
    """ Specify and generate payoff values of an Asian Call option """
    def __init__(self, BMmeasure=None, volatility=.5, start_price=30, strike_price=25):
        """
        Initialize AsianCall Integrand's'

        Args:
            BMmeasure (Measure): A BrownianMotion Measure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): K, the call/put offer
            nominal_value (int): :math:`c` such that \
                :math:`(c, \ldots, c) \in \mathcal{X}`
        """
        super().__init__(nominal_value=0)
        self.BMmeasure = BMmeasure
        self.volatility = volatility
        self.S0 = start_price
        self.K = strike_price
        self.dimFac = 0
        if not self.BMmeasure: return
        # Create a list of Asian Call Options and distribute attributes
        nBM = len(BMmeasure)
        self.fun_list = [AsianCall() for i in range(nBM)]
        self[0].BMmeasure = self.BMmeasure[0]
        self[0].dimFac = 0
        self[0].dimension = self.BMmeasure[0].dimension
        for ii in range(1,nBM): # distribute attr
            self[ii].BMmeasure = self.BMmeasure[ii]
            self[ii].dimFac = self.BMmeasure[ii].dimension/self.BMmeasure[ii-1].dimension
            self[ii].dimension = self.BMmeasure[ii].dimension

    def g(self,x,ignore):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` \
                row of an :math:`n \cdot |\mathfrak{u}|` matrix
            coord_index: set of those coordinates in sequence needed, \
                :math:`\mathfrak{u}`

        Returns:
            :math:`n \cdot p` matrix with values :math:`f(\mathbf{x}_{\mathfrak{u},i},\mathbf{c})`
            where if :math:`\mathbf{x}_i' = (x_{i, \mathfrak{u}},\mathbf{c})_j`, then :math:`x'_{ij} = x_{ij}`
            for :math:`j \in \mathfrak{u}`, and :math:`x'_{ij} = c` otherwise
        """
        SFine = self.S0*exp((-self.volatility**2/2)*self.BMmeasure.time_vector+self.volatility*x)
        AvgFine = ((self.S0/2)+SFine[:,:self.dimension-1].sum(1)+SFine[:,self.dimension-1]/2)/self.dimension
        y = maximum(AvgFine-self.K,0)
        if self.dimFac > 0:
            Scourse = SFine[:,int(self.dimFac-1)::int(self.dimFac)]
            dCourse = self.dimension/self.dimFac
            AvgCourse = ((self.S0/2)+Scourse[:,:int(dCourse)-1].sum(1)+Scourse[:,int(dCourse)-1]/2)/dCourse
            y = y-maximum(AvgCourse-self.K,0)
        return y
