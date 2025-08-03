#!/bin/sh
time find . -maxdepth 1 -name "*.ipynb" -print0 | xargs -0 -n 1 -P $(sysctl -n hw.ncpu) jupyter nbconvert --execute --to notebook --inplace --allow-errors
