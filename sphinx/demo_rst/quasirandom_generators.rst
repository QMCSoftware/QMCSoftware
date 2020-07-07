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


General Usage
-------------

.. code:: ipython3

    # Unshifted Samples
    lattice = Lattice(dimension=2, randomize=False, seed=7, backend='MPS')
    unshifted_samples = lattice.gen_samples(n_min=0,n_max=4)
    print('Shape: %s'%str(unshifted_samples.shape))
    print('Samples:\n'+str(unshifted_samples))


.. parsed-literal::

    Shape: (4, 2)
    Samples:
    [[0.   0.  ]
     [0.5  0.5 ]
     [0.25 0.25]
     [0.75 0.75]]


.. code:: ipython3

    # Shifted Samples
    lattice = Lattice(dimension=2, randomize=True, seed=7, backend='GAIL')
    shifted_samples = lattice.gen_samples(n_min=4, n_max=8)
    print('Shape: %s'%str(shifted_samples.shape))
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
    df_py = pd.read_csv('../workouts/lds_sequences/out/python_sequences.csv')
    df_py.columns = ['n',
                     'py_l_MPS','py_l_GAIL',
                     'py_s_QRNG_gc','py_s_QRNG_n','py_s_MPS_QMCPy',
                     'py_h_QRNG','py_h_Owen',
                     'py_k_QRNG']
    df_m = pd.read_csv('../workouts/lds_sequences/out/matlab_sequences.csv', header=None)
    df_m.columns = ['n', 'm_l', 'm_s','m_h']
    df_r = pd.read_csv('../workouts/lds_sequences/out/r_sequences.csv')
    df_r.columns = ['n','r_s','r_h','r_k']
    df_r.reset_index(drop=True, inplace=True)

.. code:: ipython3

    def plt_lds_comp(df,name,colors):
        fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
        labels = df.columns[1:]
        n = df['N']
        for label,color in zip(labels,colors):
            ax.loglog(n, df[label], label=label, color=color)
        ax.legend(loc='upper left')
        ax.set_xlabel('Sampling Points')
        ax.set_ylabel('Generation Time (Seconds)')
        # Metas and Export
        fig.suptitle('Speed Comparison of %s Generators'%name)

Lattice
~~~~~~~

.. code:: ipython3

    df_l = pd.concat([df_py['n'], df_py['py_l_MPS'], df_py['py_l_GAIL'],df_m['m_l']], axis=1)
    df_l.columns = ['N','QMCPy_MPS','QMCPy_GAIL','MATLAB_GAIL']
    df_l.set_index('N')




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
          <th>MATLAB_GAIL</th>
        </tr>
        <tr>
          <th>N</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>5.56e-05</td>
          <td>6.19e-05</td>
          <td>2.14e-04</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>8.00e-05</td>
          <td>1.04e-04</td>
          <td>1.38e-04</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>7.62e-05</td>
          <td>1.46e-04</td>
          <td>1.34e-04</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>1.13e-04</td>
          <td>1.78e-04</td>
          <td>1.32e-04</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>1.16e-04</td>
          <td>2.21e-04</td>
          <td>1.74e-04</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>1.47e-04</td>
          <td>2.94e-04</td>
          <td>1.54e-04</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>1.55e-04</td>
          <td>2.50e-04</td>
          <td>1.49e-04</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>1.55e-04</td>
          <td>3.23e-04</td>
          <td>1.77e-04</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>1.40e-04</td>
          <td>3.62e-04</td>
          <td>1.84e-04</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>2.11e-04</td>
          <td>3.80e-04</td>
          <td>1.89e-04</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>2.40e-04</td>
          <td>5.84e-04</td>
          <td>2.11e-04</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>2.90e-04</td>
          <td>6.17e-04</td>
          <td>2.49e-04</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>3.84e-04</td>
          <td>9.51e-04</td>
          <td>3.06e-04</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>6.89e-04</td>
          <td>1.64e-03</td>
          <td>4.60e-04</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>1.14e-03</td>
          <td>3.01e-03</td>
          <td>7.10e-04</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>2.08e-03</td>
          <td>6.09e-03</td>
          <td>1.14e-03</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>5.10e-03</td>
          <td>1.03e-02</td>
          <td>1.88e-03</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>8.62e-03</td>
          <td>1.92e-02</td>
          <td>3.76e-03</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>1.77e-02</td>
          <td>3.97e-02</td>
          <td>7.06e-03</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>2.91e-02</td>
          <td>8.40e-02</td>
          <td>1.41e-02</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    plt_lds_comp(df_l,'Lattice',colors=['r','g','b'])



.. image:: quasirandom_generators_files/quasirandom_generators_10_0.png


Sobol
~~~~~

.. code:: ipython3

    df_s = pd.concat([df_py['n'], df_py['py_s_QRNG_n'], df_py['py_s_QRNG_gc'], df_py['py_s_MPS_QMCPy'], df_m['m_s'], df_r['r_s']], axis=1)
    df_s.columns = ['N','QMCPy_QRNG_GC','QMCPy_QRNG_N','QMCPy_MPS','MATLAB','R_QRNG']
    df_s.set_index('N')




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
          <th>QMCPy_QRNG_GC</th>
          <th>QMCPy_QRNG_N</th>
          <th>QMCPy_MPS</th>
          <th>MATLAB</th>
          <th>R_QRNG</th>
        </tr>
        <tr>
          <th>N</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>1.78e-05</td>
          <td>3.19e-05</td>
          <td>2.58e-05</td>
          <td>3.13e-04</td>
          <td>3.93e-05</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>1.95e-05</td>
          <td>3.42e-05</td>
          <td>3.41e-05</td>
          <td>2.30e-04</td>
          <td>2.29e-05</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>1.84e-05</td>
          <td>2.92e-05</td>
          <td>5.18e-05</td>
          <td>2.28e-04</td>
          <td>2.26e-05</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>1.93e-05</td>
          <td>3.84e-05</td>
          <td>5.79e-05</td>
          <td>3.53e-04</td>
          <td>3.03e-05</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>1.83e-05</td>
          <td>2.83e-05</td>
          <td>8.39e-05</td>
          <td>4.88e-04</td>
          <td>3.27e-05</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>2.06e-05</td>
          <td>3.97e-05</td>
          <td>1.63e-04</td>
          <td>1.50e-03</td>
          <td>2.31e-05</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>2.16e-05</td>
          <td>4.16e-05</td>
          <td>3.22e-04</td>
          <td>4.36e-04</td>
          <td>3.62e-05</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>2.61e-05</td>
          <td>3.86e-05</td>
          <td>8.89e-04</td>
          <td>2.78e-04</td>
          <td>2.54e-05</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>3.04e-05</td>
          <td>6.34e-05</td>
          <td>1.59e-03</td>
          <td>2.29e-04</td>
          <td>3.18e-05</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>3.45e-05</td>
          <td>4.29e-05</td>
          <td>3.25e-03</td>
          <td>2.98e-04</td>
          <td>2.97e-05</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>5.13e-05</td>
          <td>6.94e-05</td>
          <td>5.10e-03</td>
          <td>2.59e-04</td>
          <td>3.26e-05</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>1.04e-04</td>
          <td>9.42e-05</td>
          <td>1.03e-02</td>
          <td>3.25e-04</td>
          <td>6.50e-05</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>1.57e-04</td>
          <td>1.83e-04</td>
          <td>1.89e-02</td>
          <td>2.97e-04</td>
          <td>6.61e-05</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>3.12e-04</td>
          <td>3.44e-04</td>
          <td>3.39e-02</td>
          <td>3.70e-04</td>
          <td>1.09e-04</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>5.17e-04</td>
          <td>5.56e-04</td>
          <td>6.89e-02</td>
          <td>4.84e-04</td>
          <td>1.96e-04</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>1.33e-03</td>
          <td>1.22e-03</td>
          <td>1.28e-01</td>
          <td>8.00e-04</td>
          <td>3.74e-04</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>1.93e-03</td>
          <td>2.23e-03</td>
          <td>2.52e-01</td>
          <td>1.30e-03</td>
          <td>1.01e-03</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>3.85e-03</td>
          <td>4.60e-03</td>
          <td>5.04e-01</td>
          <td>2.47e-03</td>
          <td>1.82e-03</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>8.26e-03</td>
          <td>9.14e-03</td>
          <td>9.99e-01</td>
          <td>4.52e-03</td>
          <td>3.78e-03</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>1.55e-02</td>
          <td>1.82e-02</td>
          <td>2.00e+00</td>
          <td>8.49e-03</td>
          <td>9.91e-03</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    plt_lds_comp(df_s,'Sobol',['r','g','b','c','m'])



.. image:: quasirandom_generators_files/quasirandom_generators_13_0.png


Halton (Generalized)
~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    df_h = pd.concat([df_py['n'], df_py['py_h_QRNG'],df_py['py_h_Owen'], df_r['r_h'],df_m['m_h']], axis=1)
    df_h.columns = ['N','QMCPy_QRNG','QMCPy_Owen','R_QRNG','MATLAB']
    df_h.set_index('N')




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
          <th>QMCPy_Owen</th>
          <th>R_QRNG</th>
          <th>MATLAB</th>
        </tr>
        <tr>
          <th>N</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>1.36e-05</td>
          <td>5.39e-04</td>
          <td>1.80e-05</td>
          <td>1.43e-04</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>1.56e-05</td>
          <td>5.52e-04</td>
          <td>1.36e-05</td>
          <td>1.05e-04</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>1.68e-05</td>
          <td>7.32e-04</td>
          <td>1.49e-05</td>
          <td>1.03e-04</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>2.03e-05</td>
          <td>6.01e-04</td>
          <td>3.32e-05</td>
          <td>1.16e-04</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>2.42e-05</td>
          <td>5.65e-04</td>
          <td>2.77e-05</td>
          <td>1.50e-04</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>4.20e-05</td>
          <td>9.39e-04</td>
          <td>3.14e-05</td>
          <td>1.14e-04</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>6.83e-05</td>
          <td>8.45e-04</td>
          <td>4.30e-05</td>
          <td>1.23e-04</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>1.39e-04</td>
          <td>1.18e-03</td>
          <td>9.00e-05</td>
          <td>1.36e-04</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>2.26e-04</td>
          <td>2.03e-03</td>
          <td>1.59e-04</td>
          <td>1.72e-04</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>4.49e-04</td>
          <td>2.74e-03</td>
          <td>2.91e-04</td>
          <td>2.45e-04</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>9.77e-04</td>
          <td>4.63e-03</td>
          <td>5.20e-04</td>
          <td>4.12e-04</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>1.69e-03</td>
          <td>7.69e-03</td>
          <td>1.07e-03</td>
          <td>7.87e-04</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>3.16e-03</td>
          <td>1.33e-02</td>
          <td>2.14e-03</td>
          <td>1.57e-03</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>6.77e-03</td>
          <td>2.54e-02</td>
          <td>4.04e-03</td>
          <td>3.13e-03</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>1.31e-02</td>
          <td>5.21e-02</td>
          <td>8.15e-03</td>
          <td>6.55e-03</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>2.68e-02</td>
          <td>9.95e-02</td>
          <td>1.81e-02</td>
          <td>1.27e-02</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>5.41e-02</td>
          <td>2.17e-01</td>
          <td>3.44e-02</td>
          <td>2.75e-02</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>1.11e-01</td>
          <td>4.42e-01</td>
          <td>7.07e-02</td>
          <td>5.76e-02</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>2.26e-01</td>
          <td>8.95e-01</td>
          <td>1.40e-01</td>
          <td>1.24e-01</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>4.59e-01</td>
          <td>1.78e+00</td>
          <td>2.89e-01</td>
          <td>2.52e-01</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    plt_lds_comp(df_h,'Halton',colors=['r','g','b','c'])



.. image:: quasirandom_generators_files/quasirandom_generators_16_0.png


Korobov
~~~~~~~

.. code:: ipython3

    df_k = pd.concat([df_py['n'], df_py['py_h_QRNG'],df_r['r_k']], axis=1)
    df_k.columns = ['N','QMCPy_QRNG','R_QRNG']
    df_k.set_index('N')




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
          <th>R_QRNG</th>
        </tr>
        <tr>
          <th>N</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2.00e+00</th>
          <td>1.36e-05</td>
          <td>2.03e-05</td>
        </tr>
        <tr>
          <th>4.00e+00</th>
          <td>1.56e-05</td>
          <td>2.08e-05</td>
        </tr>
        <tr>
          <th>8.00e+00</th>
          <td>1.68e-05</td>
          <td>1.65e-05</td>
        </tr>
        <tr>
          <th>1.60e+01</th>
          <td>2.03e-05</td>
          <td>1.99e-05</td>
        </tr>
        <tr>
          <th>3.20e+01</th>
          <td>2.42e-05</td>
          <td>2.10e-05</td>
        </tr>
        <tr>
          <th>6.40e+01</th>
          <td>4.20e-05</td>
          <td>2.69e-05</td>
        </tr>
        <tr>
          <th>1.28e+02</th>
          <td>6.83e-05</td>
          <td>1.75e-05</td>
        </tr>
        <tr>
          <th>2.56e+02</th>
          <td>1.39e-04</td>
          <td>1.79e-05</td>
        </tr>
        <tr>
          <th>5.12e+02</th>
          <td>2.26e-04</td>
          <td>1.85e-05</td>
        </tr>
        <tr>
          <th>1.02e+03</th>
          <td>4.49e-04</td>
          <td>3.37e-05</td>
        </tr>
        <tr>
          <th>2.05e+03</th>
          <td>9.77e-04</td>
          <td>3.06e-05</td>
        </tr>
        <tr>
          <th>4.10e+03</th>
          <td>1.69e-03</td>
          <td>3.71e-05</td>
        </tr>
        <tr>
          <th>8.19e+03</th>
          <td>3.16e-03</td>
          <td>5.23e-05</td>
        </tr>
        <tr>
          <th>1.64e+04</th>
          <td>6.77e-03</td>
          <td>8.42e-05</td>
        </tr>
        <tr>
          <th>3.28e+04</th>
          <td>1.31e-02</td>
          <td>3.72e-04</td>
        </tr>
        <tr>
          <th>6.55e+04</th>
          <td>2.68e-02</td>
          <td>3.94e-04</td>
        </tr>
        <tr>
          <th>1.31e+05</th>
          <td>5.41e-02</td>
          <td>8.82e-04</td>
        </tr>
        <tr>
          <th>2.62e+05</th>
          <td>1.11e-01</td>
          <td>1.42e-03</td>
        </tr>
        <tr>
          <th>5.24e+05</th>
          <td>2.26e-01</td>
          <td>3.36e-03</td>
        </tr>
        <tr>
          <th>1.05e+06</th>
          <td>4.59e-01</td>
          <td>6.72e-03</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    plt_lds_comp(df_k,'Korobov',colors=['r','g','b'])



.. image:: quasirandom_generators_files/quasirandom_generators_19_0.png


