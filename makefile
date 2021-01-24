# Makefile for, running test suites, running workouts, and compiling documentaion
#
# Syntax notes:
#   @ means don't echo command
#   2>/dev/null means hide warnings and standard error --- remove that when trying
#   to fix warnings or errors in documentation generation
#
#   -C for sphix-build will not look for conf.py
#   -b for sphinx-build will look for conf.py

tests:
	@echo "\nDoctests"
	cd qmcpy && python -m pytest --doctest-modules --disable-pytest-warnings
	@echo "\nFastests"
	python -W ignore -m unittest discover -s test/fasttests/ 1>/dev/null
	@echo "\nLongtests"
	python -W ignore -m unittest discover -s test/longtests/ 1>/dev/null

mddir = sphinx/readme_rst/
nbdir = sphinx/demo_rst/
umldir = sphinx/uml/
nbconvertcmd = jupyter nbconvert --to rst --output-dir='$(nbdir)'
SPHINXOPTS  ?= -W --keep-going
SPHINXBUILD ?= sphinx-build
SOURCEDIR = sphinx
BUILDDIR = sphinx/_build
_doc: # gets run by sphinx/conf.py so we don't need to commit files in $(mddir) and $(nbdir) 
	# Make Directries
	@-rm -r -f $(mddir) 2>/dev/null &
	@-rm -r -f $(nbdir) 2>/dev/null &
	@-rm -r -f $(umldir) 2>/dev/null &
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
	# UML Diagrams
	@mkdir $(umldir)
	#	General Uml 1
	@pyreverse -k qmcpy/ --ignore util,stopping_criterion,accumulate_data,discrete_distribution -o png 1>/dev/null && mv classes.png $(umldir)qmcpy_uml1.png
	#	General Uml 2
	@pyreverse -k qmcpy/discrete_distribution/ -o png 1>/dev/null && mv classes.png $(umldir)qmcpy_uml2.png
	#	General Uml 3
	@pyreverse -k qmcpy/ --ignore util,integrand,true_measure,discrete_distribution -o png 1>/dev/null && mv classes.png $(umldir)qmcpy_uml3.png
	#	Accumulate Data
	@pyreverse qmcpy/accumulate_data/ -o png 1>/dev/null && mv classes.png $(umldir)accumulate_data_uml.png
	#	Discrete Distribution
	@pyreverse qmcpy/discrete_distribution/ -o png 1>/dev/null && mv classes.png $(umldir)discrete_distribution_uml.png
	#	True Measure
	@pyreverse qmcpy/true_measure/ -o png 1>/dev/null && mv classes.png $(umldir)true_measure_uml.png
	#	Integrand
	@pyreverse qmcpy/integrand/ -o png 1>/dev/null && mv classes.png $(umldir)integrand_uml.png
	#	Stopping Criterion
	@pyreverse qmcpy/stopping_criterion/ -o png 1>/dev/null && mv classes.png $(umldir)stopping_criterion_uml.png
	#	Util
	@pyreverse -k qmcpy/util/ -o png 1>/dev/null && mv classes.png $(umldir)util_uml.png
	# 	Warning
	@dot -Tpng sphinx/util_warn.dot > $(umldir)util_warn.png
	#	Error
	@dot -Tpng sphinx/util_err.dot > $(umldir)util_err.png
	#	Packages
	@mv packages.png $(umldir)packages.png
doc_html: _doc
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)
doc_pdf: _doc
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR) -W --keep-going 2>/dev/null
	@cd sphinx/_build && make
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
	
