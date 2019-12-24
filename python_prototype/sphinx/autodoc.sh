#!/bin/bash
### In sphinx directory and terminal, run this file:
#  ./autodoc.sh

echo "QMCPy autodoc process starts..."

echo "$(date)"

## Use pandoc to generate HTML files from REAME.md that has LaTeX expressions

cd ../..  # to go to directory QMCSoftware
pwd
DIR=python_prototype/sphinx/rts_from_readme/
if [ ! -d $DIR ]; then
  mkdir $DIR
fi
rm DIR/*
python python_prototype/qmcpy/_util/render_readme_as_rst.py
# pandoc -s --mathjax ./README.md -o python_prototype/sphinx/rts_from_readme/QMCSoftware.html

## Generate html from notebooks
cd python_prototype

export PYTHONPATH=$PYTHONPATH:"$(pwd; cd ..)"
echo $PYTHONPATH

# run notebooks
# jupyter notebook demos/plotDemos.ipynb
cd demos
FILES=*.ipynb
for f in $FILES
do
  echo "Processing $f file..."
  jupyter-nbconvert --execute --ExecutePreprocessor.kernel_name=$CONDA_DEFAULT_ENV --ExecutePreprocessor.timeout=0 $f
done
DIR=../sphinx/html_from_demos
if [ ! -d $DIR ]; then
  mkdir $DIR
fi
rm -f $DIR/*
mv *.html $DIR
cd .. # to python_prototype

## Use sphinx to generate HTML documentation with inputs from above and
## docstrings from Python code

cd sphinx   # to return to sphinx directory

make clean;

make html

rm -fr ../../docs;  mkdir ../../docs;
cp -a _build/html/. ../../docs;

## Use sphinx to generate PDF documentation

make latex

##pandoc -s -r html _build/html/demos.html -o _build/latex/demos.tex
#pandoc -s -r html _build/html/demos.html -o _build/latex/demos.md
#pandoc _build/latex/demos.md -o _build/latex/qmcpy_demos.pdf
#pandoc -s -r html _build/html/demos.html -o _build/latex/qmcpy_demos.pdf

##pandoc -N --template=template.tex _build/latex/demos.tex -o _build/latex/demos.pdf
#pandoc -s -r html _build/html/readme.html -o _build/latex/readme.tex
##pandoc -N --template=template.tex _build/latex/readme.tex -o _build/latex/readme.pdf

cp -a _build/latex/qmcpy.pdf ../../docs/qmcpy.pdf

cp ../requirements.txt ../../docs

echo "QMCPy autodoc process ends..."