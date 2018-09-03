from fun import fun as fun
from numpy import square, cos, exp, sqrt, multiply, sum, array, cumprod, transpose, ones
from numpy.linalg import eig
from scipy.sparse import spdiags

class AsianCallFun(fun):
    volatility = 0.5
    S0 = 30
    K = 25
    T = 1
    A = []
    tVec = []
    dimFac = 1

    # Specify and generate payoff values of an Asian Call option
    def __init__(self, dimFac = []):
        super().__init__()
        if dimFac != []:
            dimVec = cumprod(dimFac, axis=0)
            nf = dimVec.size
            self = [AsianCallFun() for i in range(nf)]
            self(0).dimFac = 0
            for ii in range(nf):
                d = dimVec(ii)
                if ii > 0:
                    self[ii].dimFac = dimFac(ii - 1)
                self(ii).dimension = d
                tvec = range(d) * (self[ii].T / d)
                self(ii).tVec = tvec
                CovMat = min(transpose(tvec),tvec)
                [eigVec, eigVal] = eig(CovMat, 'vector')
                self[ii].A = multiply(sqrt(eigVal(:-1:1)), transpose(eigVec(:, :-1:1)))
        return self


    def f(self, x, coordIndex):
        # since the nominalValue = 0, this is efficient
        BM = x * self.A
        SFine = self.S0 * exp((-self.volatility ^ 2 / 2) * self.tVec + self.volatility * BM)
        AvgFine = ((self.S0 / 2) + sum(SFine(:, 1:self.dimension-1), 2) + SFine(:, self.dimension) / 2) / self.dimension
        y = max(AvgFine - self.K, 0)
        if self.dimFac > 0:
            SCoarse = SFine(:, self.dimFac: self.dimFac:end)
            dCoarse = self.dimension / self.dimFac
            AvgCoarse = ((self.S0 / 2) + sum(SCoarse(:, 1:dCoarse-1), 2) + SCoarse(:, dCoarse) / 2) / dCoarse
            y = y - max(AvgCoarse - self.K, 0)
        return y
