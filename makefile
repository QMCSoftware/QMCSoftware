# Syntax notes:
#   @ means don't echo command
#   2>/dev/null means hide warnings and standard error - remove when debugging
#   -C for sphix-build will not look for conf.py
#   -b for sphinx-build will look for conf.py

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
	@rm -r -f $(mddir) 2>/dev/null
	@rm -r -f $(nbdir) 2>/dev/null
	# READMEs --> RST
	@mkdir $(mddir)
	@grep -v  "\[\!" README.md > README2.md
	@pandoc --mathjax README2.md -o $(mddir)QMCSoftware.rst
	@rm README2.md
	# Jupyter Notebook Demos --> RST
	@mkdir $(nbdir)
	@for f in demos/*.ipynb; do \
	    echo "#\tConverting $$f"; \
	    $(nbconvertcmd) $$f 2>/dev/null;\
	done
	# Removing Colab references in rst files using regular expression
	@for f in $(nbdir)/*.rst; do \
	    grep -vE "(colab-badge.svg|Open In Colab|colab.research)" $$f > $(nbdir)/tmp.rst && mv $(nbdir)/tmp.rst  $$f; \
    done

_uml:
	# UML Diagrams
	@rm -r -f $(umldir) 2>/dev/null
	@mkdir $(umldir)
	#	Discrete Distribution Overview
	@pyreverse -k qmcpy/discrete_distribution/ -o png 1>/dev/null && mv classes.png $(umldir)discrete_distribution_overview.png
	#	True Measure Overview
	@pyreverse -k qmcpy/true_measure/ -o png 1>/dev/null && mv classes.png $(umldir)true_measure_overview.png
	#	Integrand Overview
	@pyreverse -k qmcpy/integrand/ -o png 1>/dev/null && mv classes.png $(umldir)integrand_overview.png
	#	Stopping Criterion Overview
	@pyreverse -k qmcpy/stopping_criterion/ -o png 1>/dev/null && mv classes.png $(umldir)stopping_criterion_overview.png
	#	Discrete Distribution Specific
	@pyreverse qmcpy/discrete_distribution/ -o png 1>/dev/null && mv classes.png $(umldir)discrete_distribution_specific.png
	#	True Measure Specific
	@pyreverse qmcpy/true_measure/ -o png 1>/dev/null && mv classes.png $(umldir)true_measure_specific.png
	#	Integrand Specific
	@pyreverse qmcpy/integrand/ -o png 1>/dev/null && mv classes.png $(umldir)integrand_specific.png
	#	Stopping Criterion Specific
	@pyreverse qmcpy/stopping_criterion/ -o png 1>/dev/null && mv classes.png $(umldir)stopping_criterion_specific.png
	#	Util
	@pyreverse -k qmcpy/util/ -o png 1>/dev/null && mv classes.png $(umldir)util_uml.png
	# 	Warning
	@dot -Tpng sphinx/util_warn.dot > $(umldir)util_warn.png
	#	Error
	@dot -Tpng sphinx/util_err.dot > $(umldir)util_err.png
	#	Packages
	@mv packages.png $(umldir)packages.png

doc_html: _doc _uml
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)

doc_pdf: _doc _uml
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR) -W --keep-going  2>/dev/null
	@cd sphinx/_build && make

doc_epub: _doc _uml
	@$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub

tests:
	@echo "\nDoctests"
	python -m coverage run --source=./ -m pytest --doctest-modules --disable-pytest-warnings qmcpy
	@echo "\nFastests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/fasttests/ 1>/dev/null
	@echo "\nLongtests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/longtests/ 1>/dev/null
	@echo "\nCode coverage"
	python -m coverage report -m

# "[command] | tee [logfile]" prints to both stdout and logfile
workout:
	# integration_examples
	@python workouts/integration_examples/asian_option_multi_level.py | tee workouts/integration_examples/out/asian_option_multi_level.log
	@python workouts/integration_examples/asian_option_single_level.py | tee workouts/integration_examples/out/asian_option_single_level.log
	@python workouts/integration_examples/keister.py  | tee  workouts/integration_examples/out/keister.log
	@python workouts/integration_examples/pi_problem.py | tee workouts/integration_examples/out/pi_problem.log
	# mlmc
	@python workouts/mlmc/mcqmc06.py | tee workouts/mlmc/out/mcqmc06.log
	@python workouts/mlmc/european_option.py | tee workouts/mlmc/out/european_option.log
	# lds_sequences
	@python workouts/lds_sequences/python_sequences.py | tee workouts/lds_sequences/out/python_sequences.log
	# mc_vs_qmc
	@python workouts/mc_vs_qmc/importance_sampling.py | tee workouts/mc_vs_qmc/out/importance_sampling.log
	@python workouts/mc_vs_qmc/vary_abs_tol.py | tee workouts/mc_vs_qmc/out/vary_abs_tol.log
	@python workouts/mc_vs_qmc/vary_dimension.py | tee workouts/mc_vs_qmc/out/vary_dimension.log

exportcondaenv:
	@-rm -f requirements/environment.yml 2>/dev/null &
	@conda env export --no-builds | grep -v "^prefix: " > requirements/environment.yml

exportpipreq:
	@pip freeze > requirements/dev.txt

conda_colab:
	@conda create -n qmcpy python=3.7.0
	@conda activate qmcpy
	@conda install conda-build
	@conda develop .
	@pip install -r requirements/dev.txt
	@pip install -e .

conda_doc:
	#   Assuming QMCPy repository is cloned locally and the correct branch is checked out
	@conda env create --file requirements/environment.yml
	@conda activate qmcpy
	@pip install -e .
	#   Suggest to run `make tests` to check environment is working

gen_doc:
	#   Compiling QMCPy HTML documentation.
	@make doc_html
	#   Suggest to check time stamp of index.html
	@ls -l sphinx/_build/index.html
	#   Compiling QMCPy EPUB documentation.
	@make doc_epub
	#   Suggest to check time stamp of qmcpy.epub
	@ls -l sphinx/_build/epub/qmcpy.epub
	#   Assumed pdflatex is installed.
	#   Compiling QMCPy PDF documentation. If failed, remove "2>/dev/null" in the task `doc_pdf` to debug
	@make doc_pdf
	#   Suggest to check time stamp of qmcpy.pdf
	@ls -l sphinx/_build/qmcpy.pdf