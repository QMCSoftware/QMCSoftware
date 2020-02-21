Welcome to QMCPy
================

Importing QMCPy
---------------

Here we show three different ways to import QMCPy in a Python
environment. First, we can import the package ``qmcpy`` under the alias
``qp``.

.. code:: ipython3

    import qmcpy as qp
    print(qp.name, qp.__version__)


.. parsed-literal::

    qmcpy 0.1


Alternatively, we can import individual objects from 'qmcpy' as shown
below.

.. code:: ipython3

    from qmcpy.integrand import *
    from qmcpy.true_measure import *
    from qmcpy.discrete_distribution import *
    from qmcpy.stopping_criterion import *

Lastly, we can import all objects from the package using an asterisk.

.. code:: ipython3

    from qmcpy import *

Important Notes
---------------

IID vs LDS
~~~~~~~~~~

Low discrepancy sequences (LDS) such as lattice and Sobol are not
independent like IID (independent identically distributed) points.

The code below generates 1 replication of 4 Sobol samples of 2
dimensions.

.. code:: ipython3

    distribution = Lattice(dimension=2, scramble=True, replications=0, seed=7, backend='MPS')
    distribution.gen_samples(n_min=0,n_max=4)




.. parsed-literal::

    array([[ 0.076,  0.780],
           [ 0.576,  0.280],
           [ 0.326,  0.530],
           [ 0.826,  0.030]])



Multi-Dimensional Inputs
~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want to create an integrand in QMCPy for evaluating the
following integral:

.. math:: \int_{[0,1]^d} \|x\|_2^{\|x\|_2^{1/2}} dx,

where :math:`[0,1]^d` is the unit hypercube in :math:`\mathbb{R}^d`. The
integrand is defined everywhere except at :math:`x=0` and hence the
definite integral is also defined.

The key in defining a Python function of an integrand in the QMCPy
framework is that not only it should be able to take one point
:math:`x \in \mathbb{R}^d` and return a real value, but also that it
would be able to take a set of :math:`n` sampling points as rows in a
Numpy array of size :math:`n \times d` and return an array with
:math:`n` values evaluated at each sampling point. The following
examples illustrate this point.

.. code:: ipython3

    from numpy.linalg import norm as norm
    from numpy import sqrt, array

Our first attempt maybe to create the integrand as a Python function as
follows:

.. code:: ipython3

    def f(x): return norm(x) ** sqrt(norm(x))

It looks reasonable except that maybe the Numpy function norm is
executed twice. It's okay for now. Let us quickly test if the function
behaves as expected at a point value:

.. code:: ipython3

    x = 0.01
    f(x)




.. parsed-literal::

    0.6309573444801932



What about an array that represents :math:`n=3` sampling points in a
two-dimensional domain, i.e., :math:`d=2`?

.. code:: ipython3

    x = array([[1, 0], 
               [0, 0.01],
               [0.04, 0.04]])
    f(x)




.. parsed-literal::

    1.001650000560437



Now, the function should have returned :math:`n=3` real values that
corresponding to each of the sampling points. Let's debug our Python
function.

.. code:: ipython3

    norm(x)




.. parsed-literal::

    1.0016486409914407



Numpy's ``norm(x)`` is obviously a matrix norm, but we want it to be
vector 2-norm that acts on each row of ``x``. To that end, let's add an
axis argument to the function:

.. code:: ipython3

    norm(x, axis = 1)




.. parsed-literal::

    array([ 1.000,  0.010,  0.057])



Now it's working! Let's make sure that the ``sqrt`` function is acting
on each element of the vector norm results:

.. code:: ipython3

    sqrt(norm(x, axis = 1))




.. parsed-literal::

    array([ 1.000,  0.100,  0.238])



It is. Putting everything together, we have:

.. code:: ipython3

    norm(x, axis = 1) ** sqrt(norm(x, axis = 1))




.. parsed-literal::

    array([ 1.000,  0.631,  0.505])



We have got our proper function definition now.

.. code:: ipython3

    def f(x):
        x_norms = norm(x, axis = 1)
        return x_norms ** sqrt(x_norms)

We can now create an ``integrand`` instance with our ``QuickConstruct``
class in QMCPy and then invoke QMCPy's ``integrate`` function:

.. code:: ipython3

    dim = 1
    abs_tol = .01
    distribution = IIDStdUniform(dimension=dim, seed=7)
    measure = Uniform(distribution)
    integrand = QuickConstruct(measure, custom_fun=f)
    solution,data = CLT(integrand,abs_tol=abs_tol,rel_tol=0).integrate()
    print(data)


.. parsed-literal::

    Solution: 0.6575         
    QuickConstruct (Integrand Object)
    IIDStdUniform (Discrete DiscreteDistribution Object)
    	dimension       1
    	seed            7
    	mimics          StdUniform
    Uniform (True TrueMeasure Object)
    	distrib_name    IIDStdUniform
    	lower_bound     0
    	upper_bound     1
    CLT (Stopping Criterion Object)
    	inflate         1.200
    	alpha           0.010
    	abs_tol         0.010
    	rel_tol         0
    	n_init          1024
    	n_max           10000000000
    MeanVarData (AccumulateData Object)
    	levels          1
    	solution        0.658
    	n               3305
    	n_total         4329
    	confid_int      [ 0.647  0.668]
    	time_integrate  0.002
    


For our integral, we know the true value. Let's check if QMCPy's
solution is accurate enough:

.. code:: ipython3

    true_sol = 0.658582  # In WolframAlpha: Integral[x**Sqrt[x], {x,0,1}]
    abs_tol = data.stopping_criterion.abs_tol
    qmcpy_error = abs(true_sol - solution)
    print(qmcpy_error < abs_tol)


.. parsed-literal::

    True


It's good. Shall we test the function with :math:`d=2` by simply
changing the input parameter value of dimension for QuickConstruct?

.. code:: ipython3

    dim = 2
    distribution = IIDStdUniform(dimension=dim, seed=7)
    measure = Uniform(distribution)
    integrand = QuickConstruct(measure, custom_fun=f)
    solution2,data2 = CLT(integrand,abs_tol=abs_tol,rel_tol=0).integrate()
    print(data2)


.. parsed-literal::

    Solution: 0.8309         
    QuickConstruct (Integrand Object)
    IIDStdUniform (Discrete DiscreteDistribution Object)
    	dimension       2
    	seed            7
    	mimics          StdUniform
    Uniform (True TrueMeasure Object)
    	distrib_name    IIDStdUniform
    	lower_bound     0
    	upper_bound     1
    CLT (Stopping Criterion Object)
    	inflate         1.200
    	alpha           0.010
    	abs_tol         0.010
    	rel_tol         0
    	n_init          1024
    	n_max           10000000000
    MeanVarData (AccumulateData Object)
    	levels          1
    	solution        0.831
    	n               5452
    	n_total         6476
    	confid_int      [ 0.821  0.841]
    	time_integrate  0.002
    


Once again, we could test for accuracy of QMCPy with respect to the true
value:

.. code:: ipython3

    true_sol2 = 0.827606  # In WolframAlpha: Integral[Sqrt[x**2+y**2])**Sqrt[Sqrt[x**2+y**2]], {x,0,1}, {y,0,1}]
    abs_tol2 = data2.stopping_criterion.abs_tol
    qmcpy_error2 = abs(true_sol2 - solution2)
    print(qmcpy_error2 < abs_tol2)


.. parsed-literal::

    True


