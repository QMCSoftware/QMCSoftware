import qmcpy

def integrate_default():
    sol, data = qmcpy.integrate()
    data.summarize()


integrate_default()