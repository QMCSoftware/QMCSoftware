

class QMC(object):
    
    def __init__(self, dat, dst, fun, stp):
        from DevelopOnly.Driver_QMC import currentObjs
        import os
        import sys
        import warnings
        warnings.filterwarnings("ignore")

        self.dat = dat
        self.dst = dst
        self.fun = fun
        self.stp = stp
        
        # Add all temporary paths for this package
        old_cwd = os.getcwd()
        subdir_list = ["\Accumulate_Data", "\Discrete_Distribution", "\Function", "\Stopping_Criterion", "\Tests"]
        for subdirectory in subdir_list:
            sys.path.insert(0, old_cwd + subdirectory)
        
        invalidInstance = False
        args = [self.dat, self.dst, self.fun, self.stp]
        for i in range(4):
            try:
                temp = __import__(args[i])
                args[i] = getattr(temp,args[i])
                args[i] = args[i]()
            except Exception:
                invalidInstance = True
                print("\n\nERROR: Sublcass '" + args[0] + "' not found in " + currentObjs[i][0] + " folder")
                args[i] = None
        if invalidInstance:  
            print("\nWhen inputting your own subclass for accumData, discreteDistribution, fun, or Stopping_Condition,")
            print("please name your file the same name as you class. The program will look for search for your subclass in the cooresponding directory.")
            print("\nYour accumData, Discrete_Distributuion, fun, and Stopping Class objects have been set to None")
            print("The QMC Object you created is invallid")
        self.dat = args[0]
        self.dst = args[1]
        self.fun = args[2]
        self.stp = args[3]
        #print(self.dat, self.dst, self.fun, self.stp)
        return

    def integrate(self):
        #Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
        # self.fun (fun)= an object from class fun
        # self.dst (Discrete Distribution) = an object from class discrete_distribution
        # self.stp (Spopping Criterion) = an object from class stopping_criterion

        # Initialize the accumData object and other crucial objects
        [self.dat, self.dst] = self.stp.stopYet(self.dat, self.fun, self.dst)
        while not self.dat.stage == 'done':  # the datObj.stage property tells us where we are in the process
            self.dat.updateData(self.dst, self.fun)  # compute additional data
            [self.dat, self.dst] = self.stp.stopYet(self.dat, self.fun, self.dst)  # update the status of the computation
        solution = self.dat.solution  # assign outputs
        from time import time
        self.dat.timeUsed = time() - self.dat.timeStart
        return solution, self.dat

    def integreation(self):
        pass
        
    def runDoctests(self):
        import sys
        print(sys.path)
        print("\n\n")
        import doctest

        print("meanVardata Doctests:")
        r4 = doctest.testfile("accumData/dt_meanVar.py")
        print("\n"+str(r4))
        print("----------------------------------------------------------------------------------------------------------------\n\n")
        
        print("IIDDistribution Doctests:")
        r2 = doctest.testfile("discreteDistribution/dt_IID.py")
        print("\n"+str(r2))
        print("----------------------------------------------------------------------------------------------------------------\n\n")

        print("KeisterFun Doctests:")
        r3 = doctest.testfile("fun/dt_Keister.py")
        print("\n"+str(r3))
        print("----------------------------------------------------------------------------------------------------------------\n\n")

        print("CLTStopping Doctests:")
        r1 = doctest.testfile("stoppingCriterion/dt_CLT.py")
        print("\n"+str(r1))

        

        return
    


if __name__ == "__main__":    
    # Actual Code for this section
    '''
    from Driver_QMC import userAssist
    userArgs = userAssist()
    qmc = QMC(userArgs[0], userArgs[1], userArgs[2], userArgs[3])
    '''

    # Practice
    qmc_ags = QMC('meanVarData', 'IIDDistribution', 'KeisterFun', 'CLTStopping')
    qmc_ags.runDoctests()






    


