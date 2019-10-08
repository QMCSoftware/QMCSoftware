import qmcpy

def integrate_default():
    sol, data = qmcpy.integrate()
    data.summarize()
    return sol, data


integrate_default()