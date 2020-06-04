Quasi-Random Sequence Generator Comparison
==========================================

.. code:: ipython3

    from qmcpy import *
    
    import pandas as pd
    pd.options.display.float_format = '{:.2e}'.format
    
    from numpy import *
    set_printoptions(threshold=2**10)
    set_printoptions(precision=3)
    
    from matplotlib import pyplot as plt
    import matplotlib
    %matplotlib inline
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


General Lattice & Sobol Generator Usage
---------------------------------------

The following example uses the ``Lattice`` object to generate samples.
The same code works when replacing ``Lattice`` with ``Sobol``

.. code:: ipython3

    # Unshifted Samples
    lattice = Lattice(dimension=2, scramble=False, seed=7, backend='MPS')
    unshifted_samples = lattice.gen_samples(n_min=0,n_max=4)
    print('Shape:',unshifted_samples.shape)
    print('Samples:\n'+str(unshifted_samples))


.. parsed-literal::

    Shape: (4, 2)
    Samples:
    [[0.   0.  ]
     [0.5  0.5 ]
     [0.25 0.75]
     [0.75 0.25]]


.. code:: ipython3

    # Shifted Samples
    lattice = Lattice(dimension=2, scramble=True, seed=7, backend='GAIL')
    shifted_samples = lattice.gen_samples(n_min=4, n_max=8)
    print('Shape:',shifted_samples.shape)
    print('Samples:\n'+str(shifted_samples))


.. parsed-literal::

    Shape: (4, 2)
    Samples:
    [[0.201 0.405]
     [0.701 0.905]
     [0.451 0.655]
     [0.951 0.155]]


QMCPy Generator Times Comparison
--------------------------------

Compare the speed of low-discrepancy-sequence generators from Python
(QMCPy), MATLAB, and R. The following blocks visualize a speed
comparison with MATLAB when generating 1 dimensional
unshifted/unscrambled sequences. Note that the generators are
reinitialized before every trial. <br

.. code:: ipython3

    # Load AccumulateData
    df_py = pd.read_csv('../outputs/lds_sequences/python_sequence_times.csv')
    df_py.columns = ['n','py_l_MPS','py_l_GAIL','py_s_QRNG','py_s_MPS_QMCPy']
    df_m = pd.read_csv('../outputs/lds_sequences/matlab_sequence_times.csv', header=None)
    df_m.columns = ['n', 'm_l', 'm_s']
    df_r = pd.read_csv('../outputs/lds_sequences/r_sequence_times.csv',sep=' ')
    df_r.columns = ['n','r_s']
    df_r.reset_index(drop=True, inplace=True)

.. code:: ipython3

    # Lattice DataFrame
    df_l = pd.concat([df_py['n'], df_py['py_l_MPS'], df_py['py_l_GAIL'],df_m['m_l']], axis=1)
    df_l.columns = ['N_Lattice','QMCPy_MPS','QMCPy_GAIL','MATLAB']
    df_l.set_index('N_Lattice')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>QMCPy_MPS</th>
          <th>QMCPy_GAIL</th>
          <th>MATLAB</th>
        </tr>
        <tr>
          <th>N_Lattice</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>7.71e-05</td>
          <td>8.40e-05</td>
          <td>3.76e-04</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>7.76e-05</td>
          <td>1.10e-04</td>
          <td>2.23e-04</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>1.19e-04</td>
          <td>1.31e-04</td>
          <td>1.54e-04</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>1.67e-04</td>
          <td>4.36e-04</td>
          <td>1.59e-04</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>1.52e-04</td>
          <td>2.37e-04</td>
          <td>1.68e-04</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>2.22e-04</td>
          <td>2.74e-04</td>
          <td>1.60e-04</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>1.61e-04</td>
          <td>2.60e-04</td>
          <td>1.60e-04</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>2.08e-04</td>
          <td>2.97e-04</td>
          <td>1.72e-04</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>2.44e-04</td>
          <td>4.12e-04</td>
          <td>1.87e-04</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>2.21e-04</td>
          <td>5.05e-04</td>
          <td>1.85e-04</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>3.13e-04</td>
          <td>5.84e-04</td>
          <td>2.02e-04</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>2.81e-04</td>
          <td>7.06e-04</td>
          <td>3.15e-04</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>8.63e-04</td>
          <td>9.89e-04</td>
          <td>3.59e-04</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>5.05e-04</td>
          <td>1.53e-03</td>
          <td>5.28e-04</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>8.12e-04</td>
          <td>2.69e-03</td>
          <td>6.85e-04</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>1.37e-03</td>
          <td>6.16e-03</td>
          <td>1.56e-03</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>3.42e-03</td>
          <td>9.31e-03</td>
          <td>2.35e-03</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>6.38e-03</td>
          <td>2.11e-02</td>
          <td>4.93e-03</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>1.34e-02</td>
          <td>3.67e-02</td>
          <td>9.92e-03</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>2.60e-02</td>
          <td>8.70e-02</td>
          <td>1.99e-02</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Sobol DataFrame
    df_s = pd.concat([df_py['n'], df_py['py_s_QRNG'], df_py['py_s_MPS_QMCPy'], df_m['m_s'], df_r['r_s']], axis=1)
    df_s.columns = ['N_Sobol','QMCPy_QRNG','QMCPy_MPS','MATLAB','R']
    df_s.set_index('N_Sobol')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>QMCPy_QRNG</th>
          <th>QMCPy_MPS</th>
          <th>MATLAB</th>
          <th>R</th>
        </tr>
        <tr>
          <th>N_Sobol</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>3.83e-05</td>
          <td>5.98e-04</td>
          <td>6.36e-04</td>
          <td>1.40e-04</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>3.32e-05</td>
          <td>3.82e-04</td>
          <td>3.91e-04</td>
          <td>1.69e-04</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>2.81e-05</td>
          <td>1.05e-03</td>
          <td>3.68e-04</td>
          <td>1.44e-04</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>1.42e-04</td>
          <td>1.15e-03</td>
          <td>9.57e-04</td>
          <td>1.57e-04</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>3.78e-05</td>
          <td>4.53e-04</td>
          <td>4.58e-04</td>
          <td>1.67e-04</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>5.43e-05</td>
          <td>7.07e-04</td>
          <td>5.07e-04</td>
          <td>1.48e-04</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>3.51e-05</td>
          <td>8.24e-04</td>
          <td>4.09e-04</td>
          <td>1.72e-04</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>4.73e-05</td>
          <td>9.32e-04</td>
          <td>2.84e-04</td>
          <td>1.62e-04</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>4.86e-05</td>
          <td>3.38e-03</td>
          <td>2.47e-04</td>
          <td>1.54e-04</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>6.37e-05</td>
          <td>4.27e-03</td>
          <td>2.66e-04</td>
          <td>1.96e-04</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>9.35e-05</td>
          <td>6.67e-03</td>
          <td>2.64e-04</td>
          <td>2.12e-04</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>8.11e-05</td>
          <td>1.26e-02</td>
          <td>8.80e-04</td>
          <td>2.72e-04</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>1.15e-04</td>
          <td>2.29e-02</td>
          <td>9.59e-04</td>
          <td>5.12e-04</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>2.37e-04</td>
          <td>4.16e-02</td>
          <td>1.10e-03</td>
          <td>7.29e-04</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>3.87e-04</td>
          <td>8.71e-02</td>
          <td>6.33e-04</td>
          <td>1.20e-03</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>9.27e-04</td>
          <td>1.72e-01</td>
          <td>8.61e-04</td>
          <td>2.07e-03</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>1.35e-03</td>
          <td>3.10e-01</td>
          <td>1.50e-03</td>
          <td>4.48e-03</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>2.75e-03</td>
          <td>6.05e-01</td>
          <td>2.92e-03</td>
          <td>1.42e-02</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>5.24e-03</td>
          <td>1.20e+00</td>
          <td>5.80e-03</td>
          <td>2.80e-02</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>1.15e-02</td>
          <td>2.42e+00</td>
          <td>1.11e-02</td>
          <td>7.01e-02</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # Plot AccumulateData
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    # Lattice
    n = df_l['N_Lattice']
    for l_type,color in zip(['QMCPy_MPS','QMCPy_GAIL','MATLAB'],['c','m','r']):
        ax[0].loglog(n, df_l[l_type], label=l_type, color=color)
    ax[0].legend(loc='upper left')
    ax[0].set_xlabel('Sampling Points')
    ax[0].set_ylabel('Generation Time (Seconds)')
    ax[0].set_title('Lattice')
    # Sobol Plot
    n = df_s['N_Sobol']
    for s_type,color in zip(['QMCPy_QRNG','QMCPy_MPS','MATLAB','R'],['g','y','r','k','b']):
        ax[1].loglog(n, df_s[s_type], label=s_type, color=color)
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Sampling Points')
    ax[1].set_title('Sobol')
    # Metas and Export
    fig.suptitle('Speed Comparison of Quasi-Random Sequence Generators')
    plt.savefig('../outputs/lds_sequences/lds_generator_times.png', dpi=200)



.. image:: quasirandom_generators_files/quasirandom_generators_9_0.png


For lattice, QMCPy with GAIL backend is slower than both the Magic Point
Shop backend and MATLAB. For Sobol, QMCPy with Magic Point Shop backend
is significantly slower than using PyTorch backend or generating with
MATLAB or R. It is important to note the above results are for 1
replication of unshifted/unscrambled nodes and individual generator
instances were initialized before each trial.

