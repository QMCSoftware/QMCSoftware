### In sphinx directory and terminal, run this file:
#  ./autodoc.sh


## Use pandoc to generate HTML files from REAME.md that has LaTeX expressions

cd ../..  # to go to directory QMCSoftware
python render_readme_as_html.py
# cd python_prototype
# pandoc -s --mathjax qmcpy/README.md -o ../html_from_readme/qmcpy.html

## Generate html from notebooks
cd python_prototype

# run notebooks
# jupyter notebook demos/plotDemos.ipynb
# jupyter notebook demos/qmcpy.ipynb

jupyter-nbconvert --execute --ExecutePreprocessor.kernel_name=pyqmc demos/qmcpy.ipynb --stdout >  html_from_demos/qmcpy.html
jupyter-nbconvert --execute --ExecutePreprocessor.kernel_name=pyqmc demos/plotDemos.ipynb --stdout > html_from_demos/plotDemos.html

## Use sphinx to generate HTML documentation with inputs from above and
## docstrings from Python code

cd sphinx   # to return to sphinx directory

make clean;

make html


rm -fr ../../docs; mkdir ../../docs; cp -a _build/html/ ../../docs;