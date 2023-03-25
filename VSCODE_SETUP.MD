# How to Set Up a VSCode Environment

Notes to help you get started

## Install VSCode

- Go to https://code.visualstudio.com and dowload the package
- Install the package
- Open the package

## Install miniconda
- Download package from https://docs.conda.io/en/latest/miniconda.html
- Open terminal in VSCode 
- Type `condo -h` to ensure that conda is recognized
- Close terminal in VSCode

## Set up qmcpy environment
- Open repository in VS Code
- Run commands `git checkout develop`
    - `conda env create --file environment.yml`
    - `conda activate qmcpy`
    - `pip install -e .`
- You should now see a prompt that looks like `(qmcpy) yourname@...`
- [Next time in terminal type:  `conda activate qmcpy` to get into the qmcpy environment]
- Type `Cmd P` and select `> Select Interpreter qmcpy`

## Useful extensions in VSCode
- Python
- Jupyter
- Markdown Preview Enhanced
- eps-preview, which requires
    - Postscript Language
    - pdf2svg

## To run the demos in QMCPy
- Type `jupyter notebook`
- Or type
    - `pip install jupyterlab #the first time`
    - `jupyterlab`
- For some you may need to ddownload Docker from https://www.docker.com/products/docker-desktop/
