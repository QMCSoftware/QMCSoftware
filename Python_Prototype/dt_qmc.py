"""
dt_qmc
doctest for QMC Community Software

 python dt_qmc.py -v
"""

def run_Doctests():
    import doctest
    print("CLTStopping Doctests:")
    r1 = doctest.testfile("dt_CLTStopping.py")
    print("\n"+str(r1))
    print("----------------------------------------------------------------------------------------------------------------\n\n")
    
    print("IIDDistribution Doctests:")
    r2 = doctest.testfile("dt_IIDDistribution.py")
    print("\n"+str(r2))
    print("----------------------------------------------------------------------------------------------------------------\n\n")

    print("KeisterFun Doctests:")
    r3 = doctest.testfile("dt_KeisterFun.py")
    print("\n"+str(r3))
    print("----------------------------------------------------------------------------------------------------------------\n\n")

    print("meanVardata Doctests:")
    r4 = doctest.testfile("dt_meanVardata.py")
    print("\n"+str(r4))

if __name__ == '__main__':
    run_Doctests()
