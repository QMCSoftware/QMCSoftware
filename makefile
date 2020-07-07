# Makefile for compiling QRNG, running test suites, and compiling documentaion
#
# Syntax notes:
#   @ means don't echo command
#   2>/dev/null means hide warnings and standard error --- remove that when trying
#   to fix warnings or errors in documentation generation
#
#   -C for sphix-build will not look for conf.py
#   -b for sphinx-build will look for conf.py


qrngpath = qmcpy/discrete_distribution/qrng/
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
EXT = so
endif
ifeq ($(UNAME), Darwin)
EXT = dylib
endif
ifeq ($(UNAME), Windows)
EXT = dll
endif

qrng:
	@gcc -Wall -fPIC -shared  -o $(qrngpath)qrng_lib.$(EXT) $(qrngpath)*.c -lm
	@echo Done compiling qrng C files

tests:
	@echo "\nDoctests"
	cd qmcpy && pytest --doctest-modules
	@echo "\nFastests"
	python -W ignore -m unittest discover -s test/fasttests/ 1>/dev/null
	@echo "\nLongtests"
	python -W ignore -m unittest discover -s test/longtests/ 1>/dev/null

mddir = sphinx/readme_rst/
nbdir = sphinx/demo_rst/
nbconvertcmd = jupyter nbconvert --to rst --output-dir='$(nbdir)'
SPHINXOPTS  ?= -W --keep-going
SPHINXBUILD ?= sphinx-build
SOURCEDIR = sphinx
BUILDDIR = sphinx/_build
_doc:
	# Make Directries
	@-rm -r -f $(mddir) 2>/dev/null &
	@-rm -r -f $(nbdir) 2>/dev/null &
	# READMEs --> RST
	@mkdir $(mddir)
	@grep -v  "\[\!" README.md > README2.md
	@pandoc --mathjax README2.md -o $(mddir)QMCSoftware.rst
	@rm README2.md
	@pandoc --mathjax qmcpy/README.md -o $(mddir)qmcpy.rst
	# Jupyter Notebook Demos --> RST
	@mkdir $(nbdir)
	@for f in demos/*.ipynb; do \
	echo "#\tConverting $$f"; \
	$(nbconvertcmd) $$f 2>/dev/null;\
	done
	@rm -f $(nbdir)nei_demo.rst
	@rm -r $(nbdir)nei_demo_files/
	@cd sphinx && make clean
doc_html: _doc
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)
doc_pdf: doc_html
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR) -W --keep-going 2>/dev/null
doc_epub: _doc
	@$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub
workout:
	# integration_examples
	@python workouts/integration_examples/asian_option_multi_level.py > workouts/integration_examples/out/asian_option_multi_level.log
	@python workouts/integration_examples/asian_option_single_level.py > workouts/integration_examples/out/asian_option_single_level.log
	@python workouts/integration_examples/keister.py > workouts/integration_examples/out/keister.log
	@python workouts/integration_examples/pi_problem.py > workouts/integration_examples/out/pi_problem.log
	# mlmc
	@python workouts/mlmc/mcqmc06.py > workouts/mlmc/out/mcqmc06.log
	@python workouts/mlmc/european_option.py > workouts/mlmc/out/european_option.log
	# lds_sequences
	@python workouts/lds_sequences/python_sequences.py
	# mc_vs_qmc
	@python workouts/mc_vs_qmc/importance_sampling.py
	@python workouts/mc_vs_qmc/vary_abs_tol.py
	@python workouts/mc_vs_qmc/vary_dimension.py
	
