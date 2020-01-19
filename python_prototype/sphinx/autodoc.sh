#!/bin/bash
### In sphinx directory and terminal, run this file:
#  ./autodoc.sh

echo "QMCPy autodoc process starts..."

echo "$(date)"

################################################################################
# Use pandoc to generate RST (ReStructured Text) files from REAME.md that has
# LaTeX expressions
################################################################################
cd ../..  # to go to directory QMCSoftware
pwd
cp  README.md README.bak

# remove lines in top-level README.md that contain the keywords (service status)
# in double quotes for latex compilation
grep -v  "\[\!" README.md > temp && mv temp README.md

DIR=python_prototype/sphinx/markdown_to_rst/
if [ ! -d $DIR ]; then
  mkdir $DIR
fi
rm $DIR/*
python python_prototype/sphinx/render_readme_as_rst.py
# pandoc -s --mathjax ./README.md -o python_prototype/sphinx/markdown_to_rst/QMCSoftware.html

# restore original README.md that contains certain keywords
rm README.md
mv README.bak README.md

################################################################################
# Generate RST files from Jupyter notebooks
################################################################################
cd python_prototype

export PYTHONPATH=$PYTHONPATH:"$(pwd; cd ..)"
echo $PYTHONPATH

# run notebooks
# jupyter notebook demos/plotDemos.ipynb
cd demos # In demos directory
FILES=*.ipynb
DIR=../sphinx/rst_from_demos
if [ ! -d $DIR ]; then
  mkdir $DIR
fi
rm -fr $DIR/*
DIRSUFFIX="_files"
for f in $FILES
do
  if [  "${f}" != "nei_demo.ipynb" ]; then
    echo "Processing $f file..."
    jupyter-nbconvert --execute --ExecutePreprocessor.kernel_name=$CONDA_DEFAULT_ENV --ExecutePreprocessor.timeout=0 $f --to rst
    file=${f%.ipynb}
    echo $file
  fi
done
mv *_files $DIR/
mv *.rst $DIR

cd .. # to python_prototype

################################################################################
# Use sphinx to generate HTML documentation with inputs from above and
# docstrings from Python code
################################################################################
cd sphinx   # to return to sphinx directory

make clean;

make html

rm -fr ../../docs;  mkdir ../../docs;
cp -a _build/html/. ../../docs;

################################################################################
# Use sphinx to generate PDF documentation
################################################################################
#make latex

#cp -a _build/latex/qmcpy.pdf ../../docs/qmcpy.pdf

# remove lines in requirements.txt that contain the keywords in double quotes
# for latex compilation



################################################################################
# Use sphinx to generate epub documentation
################################################################################
#make epub

#cp -a _build/epub/qmcpy.epub ../../docs/qmcpy.epub

################################################################################
# For https://qmcpy.readthedocs.io
################################################################################
grep -v "torch" ../requirements.txt > temp && mv temp ../requirements.txt

cp ../requirements.txt ../../docs

cp ../README.md ../../docs

cd ../..

git add -f docs

echo "QMCPy autodoc process ends..."
