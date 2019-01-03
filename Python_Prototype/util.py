from fun import fun
def new_qmc_problem(): # reset the list of functions
    fun.funObjs = []

def print_dict(dict):
    for key, value in dict.items():
        print("%s: %s" % (key, value))
