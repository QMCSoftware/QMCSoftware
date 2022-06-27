Creating a new conda virtual environment for QMCPy on Mac M1 or with a Python version other than 3.7
======================================================================================================

Author: Sou-Cheng Choi

Date: June 20, 2022


[QMCPy by default uses Python 3.7 for development](https://github.com/QMCSoftware/QMCSoftware/blob/master/CONTRIBUTING.md). It contains integration tests automatically executed on Windows, Mac with Intel chip, and Linux (see, for example, [a test report](`https://github.com/QMCSoftware/QMCSoftware/actions/runs/2556604393`).

The following are general steps that could be followed for creating a conda environment for QMCPy with Python version 3.x.y or maybe a relatively new architecture such as Mac with M1 chip. 

* Install Miniconda following instructions on [Latest Miniconda Installer Links](https://docs.conda.io/en/latest/miniconda.html)

    * If you are using Mac M1, use `Miniconda3 macOS Apple M1 64-bit bash` or `Miniconda3 macOS Apple M1 64-bit pkg`
  
* Create a conda virtual environment called `qmcpy` with Python version 3.9.7:

  ```
    conda create -n qmcpy python=3.9.7
  ```

* Activate the `qmcpy` environment:
 
  ```
    conda activate qmcpy
  ```
  
* Take all the packages from `requirements/dev_docs.txt` without the version specification:

    ```
    alabaster async-generator attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython ipython-genutils jedi jeepney Jinja2 jupyter jupyter-client jupyter-console keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prometheus-client prompt-toolkit ptyprocess py pydeps Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy readme-renderer recommonmark requests requests-toolbelt scipy SecretStorage Send2Trash simplegeneric six snowballstemmer Sphinx sphinx-markdown-tables sphinx-math-dollar sphinx-rtd-theme sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml stdlib-list terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
    ```

* Perform conda install on the above packages:
    
    ```
    conda install alabaster async-generator attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython ipython-genutils jedi jeepney Jinja2 jupyter jupyter-client jupyter-console keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prometheus-client prompt-toolkit ptyprocess py pydeps Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy readme-renderer recommonmark requests requests-toolbelt scipy SecretStorage Send2Trash simplegeneric six snowballstemmer Sphinx sphinx-markdown-tables sphinx-math-dollar sphinx-rtd-theme sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml stdlib-list terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
    ```

	* If the above command comes back with an error similar to the following, remove the package names in the `conda install` list.
	
	    ```    
	    PackagesNotFoundError: The following packages are not available from current channels:
	
	  - async-generator
	  - jupyter-console
	  - sphinx-math-dollar
	  - pydeps
	  - jupyter-client
	  - stdlib-list
	  - sphinx-rtd-theme
	  - prometheus-client
	  - readme-renderer
	  - ipython-genutils
	  - sphinx-markdown-tables
	  - secretstorage
	    ```
  
	* Perform conda install without the packages in the above error:

	  ```
	    conda install alabaster  attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython  jedi jeepney Jinja2 jupyter  keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prompt-toolkit ptyprocess py  Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy  recommonmark requests requests-toolbelt scipy  Send2Trash simplegeneric six snowballstemmer Sphinx    sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml  terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
	  ```
       
	* Issue `pip install` for the packages in the error message:

	  ```
	    pip install  pip install async-generator jupyter-console sphinx-math-dollar pydeps jupyter-client stdlib-list sphinx-rtd-theme prometheus-client readme-renderer ipython-genutils sphinx-markdown-tables secretstorage
	  ```
  
* Issue the following command to compile C and C++ files in QMCPy:
    
  ```
    pip install -e .
  ```
* Lastly, run tests to make sure the environment is working:

  ```
    make tests
  ```
  
* Capture the Python environment in a `.yml` file in `requirements/` for future use:

  ```
    conda env export --no-builds | grep -v "^prefix: " > requirements/environment_mac_m1_python_3_9_7.yml
  ```