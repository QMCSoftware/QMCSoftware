import os
from ..util import ParameterError
from numpy import *


def latnetbuilder_linker(lnb_dir ='./', out_dir='./', fout_prefix='lnb4qmcpy'):
    """
    >>> from ..discrete_distribution import Lattice,DigitalNetB2
    >>> _ = os.system('docker run --name lnb -dit -p 4242:4242 umontrealsimul/latnetbuilder:light > /dev/null')
    >>> _ = os.system('docker exec -it lnb latnetbuilder -t lattice -c ordinary -s 2^16 -d 100 -f CU:P2 -q 2 -w product:0.1 -e fast-CBC -o lnb.lattice')
    >>> _ = os.system('docker cp lnb:lnb.lattice/ ./')
    >>> lnb_qmcpy_f = latnetbuilder_linker(lnb_dir='./lnb.lattice/',out_dir='./lnb.lattice',fout_prefix='mylattice')
    >>> lat = Lattice(dimension = 5,randomize=False,order='linear',generating_vector=lnb_qmcpy_f)
    >>> lat.gen_samples(8,warn=False)
    array([[0.   , 0.   , 0.   , 0.   , 0.   ],
           [0.125, 0.875, 0.625, 0.875, 0.125],
           [0.25 , 0.75 , 0.25 , 0.75 , 0.25 ],
           [0.375, 0.625, 0.875, 0.625, 0.375],
           [0.5  , 0.5  , 0.5  , 0.5  , 0.5  ],
           [0.625, 0.375, 0.125, 0.375, 0.625],
           [0.75 , 0.25 , 0.75 , 0.25 , 0.75 ],
           [0.875, 0.125, 0.375, 0.125, 0.875]])
    >>> _ = os.system('docker rm -f lnb > /dev/null')
    
    #>>> _ = os.system('docker exec -it lnb latnetbuilder -t net -c sobol -s 2^16 -d 10 -f projdep:t-value -q inf -w order-dependent:0:0,1,1 -e random-CBC:70 -o lnb.net')
    #>>> _ = os.system('docker cp lnb:lnb.net/ ./')

    Args:
        lnb_dir (str): relative path to directory where `outputMachine.txt` is stored 
            e.g. 'my_lnb/poly_lat/'
        out_dir (str): relative path to directory where output should be stored
            e.g. 'my_lnb/poly_lat_qmcpy/'
        fout_prefix (str): start of output file name. 
            e.g. 'my_poly_lat_vec' 
    
    Return:
        str: path to file which can be passed into QMCPy's Lattice or Sobol' in order to use 
             the linked latnetbuilder generating vector/matrix
             e.g. 'my_poly_lat_vec.10.16.npy'
    
    Adapted from latnetbuilder parser:
        https://github.com/umontreal-simul/latnetbuilder/blob/master/python-wrapper/latnetbuilder/parse_output.py#L84
    """
    with open(lnb_dir+'/output.txt') as f:
        Lines = f.read().split("\n")
    sep = '   #'
    if 'ordinary' in Lines[0]:
        dim = int(Lines[3].split(sep)[0].strip())
        nb_points = int(Lines[4].split(sep)[0].strip())
        gen_vector = array([int(Lines[6+i].strip()) for i in range(dim)],dtype=uint64)
        f_out = '%s/%s.%d.%d.npy'%(out_dir,fout_prefix,dim,log2(nb_points))
        save(f_out,gen_vector)
        return f_out
    elif 'polynomial' in Lines[0]:
        raise ParameterError("latnetbuilder polynomail lattice not yet supported in QMCPy.")
    elif 'sobol' in Lines[0]:
        raise ParameterError("latnetbuilder sobol not yet supported in QMCPy.")
    elif 'explicit' in Lines[0]:
        raise ParameterError("latnetbuilder explicit construciton not yet supported in QMCPy.")
    else:
        raise ParameterError("output.txt expected to have 'ordinary', 'polynomial', 'sobol', or 'explicit' in first line.")

    