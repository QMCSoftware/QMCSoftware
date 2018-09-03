from fun import fun as fun
from numpy import square, cos, exp, sqrt, multiply, sum, array, cumprod, transpose, ones
from numpy.linalg import eig
from scipy.sparse import spdiags

class AsianCallFun(fun):

    # Specify and generate payoff values of an Asian Call option
    def __init__(self, dimFac = []):
        self.volatility = 0.5
        self.S0 = 30
        self.K = 25
        self.T = 1
        self.A = []
        self.tVec = []
        if dimFac == []:
            self.dimFac = 1
        else:
            self.dimVec = cumprod(dimFac, axis=0)
            nf = self.dimVec.size
            acf_array = [AsianCallFun() for i in range(nf)]
            acf_array[0].dimFac = 0
            for ii in range(nf):
                d = self.dimVec(ii)
                if ii > 0:
                    acf_array[ii].dimFac = dimFac(ii - 1)
                acf_array[ii].dimension = d
                tvec = range(d) * (acf_array[ii].T / d)
                acf_array(ii).tVec = tvec
                CovMat = min(transpose(tvec),tvec)
                [eigVec, eigVal] = eig(CovMat, 'vector')
                acf_array[ii].A = multiply(sqrt(eigVal[-1:-1:1]), transpose(eigVec[:,-1:-1:1]))

            self = acf_array
        super().__init__()

    def f(self, x, coordIndex):
        # since the nominalValue = 0, this is efficient
        BM = multiply(x, self.A)
        SFine = self.S0 * exp((-self.volatility ^ 2 / 2) * self.tVec + self.volatility * BM)
        AvgFine = ((self.S0 / 2) + sum(SFine[:, 1:self.dimension-1], 2) + SFine[:, self.dimension] / 2) / self.dimension
        y = max(AvgFine - self.K, 0)
        if self.dimFac > 0:
            SCoarse = SFine[:, self.dimFac: self.dimFac:-1]
            dCoarse = self.dimension / self.dimFac
            AvgCoarse = ((self.S0 / 2) + sum(SCoarse[:, 1:dCoarse-1], 2) + SCoarse[:, dCoarse] / 2) / dCoarse
            y = y - max(AvgCoarse - self.K, 0)
        return y
