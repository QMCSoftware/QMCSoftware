### In sphinx directory and terminal, run this file:
#  ./autodoc.sh


## Use pandoc to generate HTML files from REAME.md that has LaTeX expressions

cd ../..  # to go to directory QMCSoftware
python render_readme_as_html.py
# cd python_prototype
# pandoc -s --mathjax qmcpy/README.md -o ../html_from_readme/qmcpy.html


## Use sphinx to generate HTML documentation with inputs from above and
## docstrings from Python code

cd python_prototype/sphinx   # to return to sphinx directory
make clean; make html
rm -fr ../../docs; mkdir ../../docs; cp -a _build/html/ ../../docs;