class Keister(Integrand):
    def g(self, x):
        dimension = x.shape[1]
        normx = norm(x, 2, axis=1)
        y = pi ** (dimension / 2.0) * cos(normx)
        return y
integrand = Keister()