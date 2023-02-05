import setuptools
from setuptools import Extension
from setuptools.command.install import install
from setuptools import Command
import os


class CustomInstall(install):
    """Custom handler for the 'install' command."""

    def run(self):
        # compile c files
        try:
            os.system('pip install -e .')
        except:
            print('Problem installing qmcpy')
        # compile files used for documentation
        try:
            os.system('make _doc')
        except:
            print('Problem compiling html or pdf documentation')
        super(CustomInstall, self).run()


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./qmcpy/qmcpy.egg-info")


try:
    with open("README.md", "r", encoding="utf-8", errors='ignore') as fh:
        long_description = fh.read()
except:
    long_description = "QMCPy"

packages = [
    'qmcpy',
    'qmcpy.discrete_distribution',
    'qmcpy.true_measure',
    'qmcpy.integrand',
    'qmcpy.stopping_criterion',
    'qmcpy.accumulate_data',
    'qmcpy.machine_learning',
    'qmcpy.util',
    'qmcpy.discrete_distribution.lattice',
    'qmcpy.discrete_distribution.c_lib',
    'qmcpy.discrete_distribution.digital_net_b2',
    'qmcpy.machine_learning.c_lib',
    'qmcpy.machine_learning.compression']

setuptools.setup(
    name="qmcpy",
    version="1.3.2",
    author="Fred Hickernell, Sou-Cheng T. Choi, Mike McCourt, Jagadeeswaran Rathinavel, Aleksei Sorokin",
    author_email="asorokin@hawk.iit.edu",
    license='Apache license 2.0',
    description="(Quasi) Monte Carlo Framework in Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://qmcsoftware.github.io/QMCSoftware/",
    download_url="https://github.com/QMCSoftware/QMCSoftware/releases/tag/v0.4.gz",
    packages=packages,
    install_requires=[
        'numpy >= 1.17.0',
        'scipy >= 1.0.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords=['quasi', 'monte', 'carlo', 'community', 'software', 'cubature', 'numerical', 'integration', 'discrepancy',
              'sobol', 'lattice'],
    python_requires=">=3.5",
    include_package_data=True,
    ext_modules=[
        Extension(
            name='qmcpy.discrete_distribution.c_lib.c_lib',
            sources=[
                'qmcpy/discrete_distribution/c_lib/halton_qrng.c',
                'qmcpy/discrete_distribution/c_lib/digital_net_b2.c',
                'qmcpy/discrete_distribution/c_lib/fwht.c', ], ),
        Extension(
            name='qmcpy.machine_learning.c_lib.c_lib',
            sources=[
                'qmcpy/machine_learning/c_lib/computeMXY.c',
                #'qmcpy/machine_learning/c_lib/computeMXYmex.c',

            ], ),
                 ],
    cmdclass={
        'clean': CleanCommand,
        'install': CustomInstall})
