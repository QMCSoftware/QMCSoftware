import setuptools
from setuptools import Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = [
    'qmcpy',
    'qmcpy.true_measure',
    'qmcpy.stopping_criterion',
    'qmcpy.discrete_distribution',
    'qmcpy.accumulate_data',
    'qmcpy.util',
    'qmcpy.integrand',
    'qmcpy.discrete_distribution.lattice',
    'qmcpy.discrete_distribution.qrng',
    'qmcpy.discrete_distribution.sobol']

setuptools.setup(
    name="qmcpy",
    version="0.2",
    author="Fred Hickernell, Sou-Cheng T. Choi, Mike McCourt, Jagadeeswaran Rathinavel, Aleksei Sorokin",
    author_email="asorokin@hawk.iit.edu",
    license='Apache license 2.0',
    description="(Quasi) Monte Carlo Framework in Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qmcsoftware.github.io/QMCSoftware/",
    download_url="https://github.com/QMCSoftware/QMCSoftware/archive/v0.1-alpha.tar.gz",
    packages=packages,
    install_requires=[
        'scipy >= 1.2.0',
        'numpy >= 1.18.5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords=['quasi','monte','carlo','community','software','cubature','numerical','integration','discrepancy','sobol','lattice'],
    python_requires=">=3.5",
    ext_modules=[
        Extension(
            name='qmcpy.discrete_distribution.qrng.qrng_lib',
            sources=['qmcpy/discrete_distribution/qrng/ghalton.c',
                    'qmcpy/discrete_distribution/qrng/korobov.c',
                    'qmcpy/discrete_distribution/qrng/MRG63k3a.c',
                    'qmcpy/discrete_distribution/qrng/sobol.c'],
            extra_compile_args=['-fPIC','-shared','-lm'])]
)
