#!/bin/sh
#This is running all notebooks and timing the output
time jupyter nbconvert --execute --to notebook --inplace *.ipynb --allow-errors

