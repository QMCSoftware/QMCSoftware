'''
To install: open a terminal and in directory QMCSoftware, run:

    python setup.py install
    python setup.py clean

This will put qmcpy in  ~/anaconda3/envs/qmcpy/lib/python3.7/site-packages/

To uninstall, do the following:

    pip uninstall qmcpy
'''

import setuptools
from setuptools.command.install import install
from setuptools import Command
import os
import subprocess

class CustomInstall(install):
    """Custom handler for the 'install' command."""
    def run(self):
        try:
            subprocess.check_call('make qrng',shell=True)
        except:
            print('Problem compiling qrng c files')
        super().run()

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./qmcpy/qmcpy.egg-info")

long_description = "Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have five main components: an integrand, a true measure, a discrete distribution, a stopping criterion, and summary output data. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above five components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the five kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications."

setuptools.setup(
    name="qmcpy",
    version="0.1",
    author="Fred Hickernell, Sou-Cheng T. Choi, Mike McCourt, Jagadeeswaran Rathinavel, Aleksei Sorokin",
    author_email="asorokin@hawk.iit.edu",
    description="(Quasi) Monte Carlo Framework in Python 3.6 and 3.7",
    long_description=long_description,
    long_description_content_type="text",
    url="https://github.com/QMCSoftware/QMCSoftware",
    package_dir={"": "qmcpy"},
    packages=setuptools.find_packages('qmcpy'),
    install_requires=['numpy','scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: IIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    cmdclass={
        'clean': CleanCommand,
        'install': CustomInstall}
)
