QMCPy for Lebesgue Integration
==============================

This notebook will give examples of how to use QMCPy for integration
problems that not are defined in terms of a standard measure. i.e.
Uniform or Gaussian.

.. code:: ipython3

    from qmcpy import *
    from numpy import *

Sample Problem 1
----------------

:math:`y = \int_{[0,2]} x^2 dx, \:\: \mbox{Lebesgue Measure}`

:math:`\phantom{y} = 2\int_{[0,2]} \frac{x^2}{2} dx, \:\: \mbox{Uniform Measure}`

.. code:: ipython3

    abs_tol = .01
    dim = 1
    a = 0
    b = 2
    true_value = 8./3

.. code:: ipython3

    # Lebesgue Measure
    distribution = IIDStdUniform(dim, seed=7)
    measure = Lebesgue(distribution, lower_bound=a, upper_bound=b)
    integrand = CustomFun(measure, lambda x: x**2)
    solution,data = CubMCCLT(integrand, abs_tol=abs_tol).integrate()
    print('y = %.3f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = 2.669
    Within tolerance: True


.. code:: ipython3

    # Uniform Measure
    distribution = IIDStdUniform(dim, seed=7)
    measure = Uniform(distribution, lower_bound=a, upper_bound=b)
    integrand = CustomFun(measure, lambda x: 2*(x**2))
    solution,data = CubMCCLT(integrand, abs_tol=abs_tol).integrate()
    print('y = %.3f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = 2.669
    Within tolerance: True


Sample Problem 2
----------------

:math:`y = \int_{[a,b]^d} ||x||_2^2 dx, \:\: \mbox{Lebesgue Measure}`

:math:`\phantom{y} = \Pi_{i=1}^d (b_i-a_i)\int_{[a,b]^d} ||x||_2^2 \; [ \Pi_{i=1}^d (b_i-a_i)]^{-1} dx, \:\: \mbox{Uniform Measure}`

.. code:: ipython3

    abs_tol = .001
    dim = 2
    a = array([1.,2.])
    b = array([2.,4.])
    true_value = ((a[0]**3-b[0]**3)*(a[1]-b[1])+(a[0]-b[0])*(a[1]**3-b[1]**3))/3
    print('Answer = %.5f'%true_value)


.. parsed-literal::

    Answer = 23.33333


.. code:: ipython3

    # Lebesgue Measure
    distribution = Sobol(dim, randomize=True, seed=7, backend='QRNG')
    measure = Lebesgue(distribution, lower_bound=a, upper_bound=b)
    integrand = CustomFun(measure, lambda x: (x**2).sum(1))
    solution,data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
    print('y = %.5f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = 23.33343
    Within tolerance: True


.. code:: ipython3

    # Uniform Measure
    distribution = Sobol(dim, randomize=True, seed=17, backend='QRNG')
    measure = Uniform(distribution, lower_bound=a, upper_bound=b)
    integrand = CustomFun(measure, lambda x: (b-a).prod()*(x**2).sum(1))
    solution,data = CubQMCCLT(integrand, abs_tol=abs_tol).integrate()
    print('y = %.5f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = 23.33318
    Within tolerance: True


Sample Problem 3
----------------

Integral that cannot be done in terms of any standard mathematical
functions
`(WOLFRAM) <https://reference.wolfram.com/language/tutorial/IntegralsThatCanAndCannotBeDone.html>`__\ 

.. math:: y = \int_{[a,b]} \frac{\sin{x}}{\log{x}} dx, \:\: \mbox{Lebesgue Measure}

Mathematica Code: ``Integrate[Sin[x]/Log[x], {x,a,b}]``

.. code:: ipython3

    abs_tol = .0001
    dim = 1
    a = 3
    b = 5
    true_value = -0.87961 

.. code:: ipython3

    # Lebesgue Measure
    distribution = Lattice(dim, randomize=True, seed=7, backend='GAIL')
    measure = Lebesgue(distribution, lower_bound=a, upper_bound=b)
    integrand = CustomFun(measure, lambda x: sin(x)/log(x))
    solution,data = CubQMCLatticeG(integrand, abs_tol=abs_tol).integrate()
    print('y = %.3f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = -0.880
    Within tolerance: True


Sample Problem 4
----------------

Integral over :math:`\mathbb{R}^d`

.. math:: y = \int_{\mathbb{R}^2} e^{-||x||_2^2} dx

.. code:: ipython3

    abs_tol = .1
    dim = 2
    true_value = pi

.. code:: ipython3

    distribution = Lattice(dim)
    measure = Lebesgue(distribution, lower_bound=-inf, upper_bound=inf)
    integrand = CustomFun(measure, lambda x: exp(-x**2).prod(1))
    solution,data = CubQMCLatticeG(integrand,abs_tol=abs_tol).integrate()
    print('y = %.3f'%solution)
    print('Within tolerance: %s'%(abs((solution-true_value))<abs_tol))


.. parsed-literal::

    y = 3.142
    Within tolerance: True


