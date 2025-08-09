from ..util import _univ_repr

class Data(object):

    def __init__(self, parameters):
        self.parameters = parameters

    def __repr__(self):
        string = _univ_repr(self, 'Data', self.parameters + ['time_integrate'])
        if hasattr(self,"stopping_crit") and self.stopping_crit:
            string += '\n'+str(self.stopping_crit)
        if hasattr(self,"integrand") and self.integrand:
            string += '\n'+str(self.integrand)
        if hasattr(self,"true_measure") and self.true_measure:
            string += '\n'+str(self.true_measure)
        if hasattr(self,"discrete_distrib") and self.discrete_distrib:
            string += '\n'+str(self.discrete_distrib)
        return string
