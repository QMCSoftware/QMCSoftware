import argparse
import sys

from QMC import QMC

currentObjs = [
    ["Accumulate_Data", ("meanVar",)],
    ["Discrete_Distribution", ("IID",)], 
    ["Function", ("AsianCall","Keister")], 
    ["Stopping_Condition",("CLT",)]
    ]
                
def getCurrentObjects():
    currentObjects_String = "QMC:\n"
    for k in range(len(currentObjs)):
        currentObjects_String += ("\t" + currentObjs[k][0]+"\n")
        for subClass in currentObjs[k][1]:
            currentObjects_String += "\t\t" + subClass + "\n"
    return currentObjects_String

def userAssist():
    currentObjects_String = getCurrentObjects()
    print(currentObjects_String)       
    print("Select an Accumulated Data, Discrete Distributuion, Funciton, and Stopping Criterion Objects")
    print("For example, you may input 'meanVar, IID, Keister, CLT'")
    userInput = input("QMC >>> ")
    print()
    userArgs = ()
    try:
        userArgs = tuple(userInput.replace(" ", "").split(","))
    except Exception:
        print("Please input arguments in the form 'meanVar, IID, Keister,CLT'")
        return
    
    if len(userArgs) != 4:
        print("Please input 4 arguments in the form 'meanVar, IID, Keister,CLT'")
        return
    
    return userArgs
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Beginning Argparse")

        parser = argparse.ArgumentParser()
        parser.add_argument('--a', '--Accumulated Data', type= str, default = "meanVar", help= str(currentObjs[0][1]), metavar='a')
        parser.add_argument('--d', '--Discrete Distribution', type=str, default = "IID", help= str(currentObjs[1][1]), metavar='d')
        parser.add_argument('--f', '--Function', type=str, default = "Keister", help = str(currentObjs[2][1]), metavar='f')
        parser.add_argument('--s', '--Stopping Criterion', type=str, default = "CLT", help = str(currentObjs[3][1]), metavar='s')

        args = parser.parse_args()
        dat=args.a
        dst=args.d
        fun=args.f
        stp=args.s
        #print(dat,dst,fun,stp)
        qmc = QMC(dat,dst,fun,stp)
        

    else:
        userArgs = userAssist()
        #print(userArgs)
        qmc = QMC(userArgs[0], userArgs[1], userArgs[2], userArgs[3])

