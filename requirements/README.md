How to create a new conda environment for QMCPy with a certain Python version other than 3.7?
=============================================================================================

Author: Sou-Cheng Choi
Date: June 20, 2022

The following are general steps that could be followed for creating a conda environment for QMCPy with Python version 3.x.y and architecture (e.g., Mac M1).

* Install Miniconda following instructions on `https://docs.conda.io/en/latest/miniconda.html`

    * If you are using Mac M1, use `Miniconda3 macOS Apple M1 64-bit...`
  
* Create a conda environment
  ```
    conda create -n qmcpy python=3.9.7
  ```

* Activate the above environment
  ```
    conda activate qmcpy
  ```
* Take all the packages from requirements/dev_docs.txt without `==[version]`:

    ```
    alabaster async-generator attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython ipython-genutils jedi jeepney Jinja2 jupyter jupyter-client jupyter-console keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prometheus-client prompt-toolkit ptyprocess py pydeps Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy readme-renderer recommonmark requests requests-toolbelt scipy SecretStorage Send2Trash simplegeneric six snowballstemmer Sphinx sphinx-markdown-tables sphinx-math-dollar sphinx-rtd-theme sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml stdlib-list terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
    ```

* Perform conda install on the above packages:
    
    ```
    conda install alabaster async-generator attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython ipython-genutils jedi jeepney Jinja2 jupyter jupyter-client jupyter-console keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prometheus-client prompt-toolkit ptyprocess py pydeps Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy readme-renderer recommonmark requests requests-toolbelt scipy SecretStorage Send2Trash simplegeneric six snowballstemmer Sphinx sphinx-markdown-tables sphinx-math-dollar sphinx-rtd-theme sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml stdlib-list terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
    ```

* If the above command comes back with  the following error, remove the package names in the error list 

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
  
* Perform conda install without the above error packages:

  ```
    conda install alabaster  attrs Babel backcall bleach certifi cffi chardet codecov commonmark coverage cryptography cycler dataclasses decorator defusedxml docutils future idna imagesize importlib-metadata iniconfig ipykernel ipython  jedi jeepney Jinja2 jupyter  keyring kiwisolver Markdown markupsafe matplotlib more-itertools nbconvert nbformat notebook numpy pandas pandoc pandocfilters parso pkginfo pluggy ply prompt-toolkit ptyprocess py  Pygments pylint pyparsing pyrsistent pytest python-dateutil pytz pyzmq qtconsole QtPy  recommonmark requests requests-toolbelt scipy  Send2Trash simplegeneric six snowballstemmer Sphinx    sphinxcontrib-applehelp sphinxcontrib-devhelp sphinxcontrib-htmlhelp sphinxcontrib-jsmath sphinxcontrib-qthelp sphinxcontrib-serializinghtml  terminado testpath toml tornado tqdm traitlets twine typing-extensions urllib3 wcwidth webencodings zipp 
  ```
       
* Issue pip install for the error packages:

  ```
    pip install  pip install async-generator jupyter-console sphinx-math-dollar pydeps jupyter-client stdlib-list sphinx-rtd-theme prometheus-client readme-renderer ipython-genutils sphinx-markdown-tables secretstorage
  ```
  
* Issue the following to compile c and cpp files in QMCPy:
    
  ```
    pip install -e .
  ```
* Lastly, run tests to make sure the environment is working

  ```
    make tests
  ```
  
* Capture the Python environment in a .yml file in `requirements/`:

  ```
    conda env export --no-builds | grep -v "^prefix: " > requirements/environment_mac_m1_python_3_9_7.yml
  ```