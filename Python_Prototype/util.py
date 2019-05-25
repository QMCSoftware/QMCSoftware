from fun import fun
from discreteDistribution import discreteDistribution
def new_qmc_problem(): # reset class lists
    fun.funObjs = []
    discreteDistribution.distribObjs = []

def print_dict(dict):
    for key, value in dict.items():
        print("%s: %s" % (key, value))
