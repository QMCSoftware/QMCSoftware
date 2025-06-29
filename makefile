# Syntax notes:
#   @ means don't echo command
#   2>/dev/null means hide warnings and standard error – remove when debugging
#   -C for sphinx-build will not look for conf.py
#   -b for sphinx-build will look for conf.py

mddir          = sphinx/md_rst/
nbdir          = sphinx/demo_rst/
umldir         = sphinx/uml/
nbconvertcmd   = jupyter nbconvert --to rst --output-dir='$(nbdir)'
SPHINXOPTS    ?= -W --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR      = sphinx
BUILDDIR       = sphinx/_build

# --------------------------------------------------------------------------- #
# Declare phony targets so they’re not mistaken for files                     #
# --------------------------------------------------------------------------- #
.PHONY: _doc _uml doc_html doc_pdf doc_epub \
        doctests doctests_no_docker fasttests longtests coverage \
        tests tests_no_docker workout export_conda_env conda_env \
        rm_qmcpy_env gen_doc \
        fasttest_parallel longtest_parallel doctest_parallel \
        tests_parallel

# --------------------------------------------------------------------------- #
# Documentation helpers                                                      #
# --------------------------------------------------------------------------- #
_doc:  # gets run by sphinx/conf.py so we don't need to commit files in $(mddir) and $(nbdir)
	# Make Directories
	@rm -r -f $(mddir) 2>/dev/null
	@rm -r -f $(nbdir) 2>/dev/null
	# READMEs --> RST
	@mkdir $(mddir)
	@grep -v  "\[\!" README.md > README2.md
	@pandoc --mathjax README2.md -o $(mddir)QMCSoftware.rst
	@rm README2.md
	# CONTRIBUTING --> RST
	@pandoc --mathjax CONTRIBUTING.md -o $(mddir)CONTRIBUTING.rst
	# Jupyter Notebook Demos --> RST
	@mkdir $(nbdir)
	@for f in demos/*.ipynb; do \
		echo "#\tConverting $$f"; \
		$(nbconvertcmd) $$f 2>/dev/null; \
	done
	@for f in demos/*/*/*.ipynb; do \
		echo "#\tConverting $$f"; \
		$(nbconvertcmd) $$f 2>/dev/null; \
	done
	# Removing Colab references in rst files using regular expression
	@for f in $(nbdir)/*.rst; do \
		grep -vE "(colab-badge.svg|Open In Colab|colab.research)" $$f > $(nbdir)/tmp.rst && mv $(nbdir)/tmp.rst $$f; \
	done
	# Add slash to * for "**kwargs"
	find . -name "*.rst" -type f -exec sed -i.bak 's/\*\*kwargs/\\*\\*kwargs/g' {} + && \
	find . -name "*.rst.bak" -delete

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
	@cd $(BUILDDIR) && rm -f qmcpy.pdf
	@$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR) -W --keep-going 2>/dev/null
	@cd $(BUILDDIR) && $(MAKE)

doc_epub: _doc _uml
	@$(SPHINXBUILD) -b epub $(SOURCEDIR) $(BUILDDIR)/epub

# --------------------------------------------------------------------------- #
# Serial test targets                                                         #
# --------------------------------------------------------------------------- #
doctests:
	@echo "\nDoctests"
	python -m coverage run --source=./qmcpy/ -m pytest --doctest-modules --disable-pytest-warnings qmcpy

doctests_no_docker:
	@echo "\nDoctests Without Docker Containers"
	python -m coverage run --source=./qmcpy/ -m pytest --doctest-modules \
		--ignore qmcpy/integrand/um_bridge_wrapper.py --disable-pytest-warnings qmcpy

fasttests:
	@echo "\nFasttests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/fasttests/ 1>/dev/null

longtests:
	@echo "\nLongtests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/longtests/ 1>/dev/null

coverage:
	@echo "\nCode coverage"
	python -m coverage report -m

tests: doctests fasttests longtests coverage
tests_no_docker: doctests_no_docker fasttests longtests coverage

# --------------------------------------------------------------------------- #
# Parallel test targets (pytest-xdist)                                        #
# --------------------------------------------------------------------------- #
fasttest_parallel:
	python -m pytest -q -W ignore::Warning -n auto \
		--cov=. --cov-append test/fasttests/

longtest_parallel:
	python -m pytest -q -W ignore::Warning -n auto \
		--cov=. --cov-append test/longtests/

doctest_parallel:
	python -m pytest -q -W ignore::Warning -n auto \
		--cov=qmcpy --cov-append --doctest-modules \
		--ignore=qmcpy/integrand/um_bridge_wrapper.py qmcpy

# Aggregate: run all three parallel suites
tests_parallel: fasttest_parallel longtest_parallel doctest_parallel

# --------------------------------------------------------------------------- #
# Miscellaneous utility targets                                               #
# --------------------------------------------------------------------------- #
workout:
	# integration_examples
	@python workouts/integration_examples/asian_option_multi_level.py     | tee workouts/integration_examples/out/asian_option_multi_level.log
	@python workouts/integration_examples/asian_option_single_level.py    | tee workouts/integration_examples/out/asian_option_single_level.log
	@python workouts/integration_examples/keister.py                      | tee workouts/integration_examples/out/keister.log
	@python workouts/integration_examples/pi_problem.py                   | tee workouts/integration_examples/out/pi_problem.log
	# mlmc
	@python workouts/mlmc/mcqmc06.py        | tee workouts/mlmc/out/mcqmc06.log
	@python workouts/mlmc/european_option.py | tee workouts/mlmc/out/european_option.log
	# lds_sequences
	@python workouts/lds_sequences/python_sequences.py | tee workouts/lds_sequences/out/python_sequences.log
	# mc_vs_qmc
	@python workouts/mc_vs_qmc/importance_sampling.py | tee workouts/mc_vs_qmc/out/importance_sampling.log
	@python workouts/mc_vs_qmc/vary_abs_tol.py        | tee workouts/mc_vs_qmc/out/vary_abs_tol.log
	@python workouts/mc_vs_qmc/vary_dimension.py      | tee workouts/mc_vs_qmc/out/vary_dimension.log

export_conda_env:
	@-rm -f environment.yml 2>/dev/null &
	@conda env export --no-builds | grep -v "^prefix: " > environment.yml

conda_env:
	# Assuming QMCPy repository is cloned locally and the correct branch is checked out
	@conda env create --file environment.yml
	@conda activate qmcpy
	@pip install -e .
	# Suggest to run `make tests` to check environment is working

rm_qmcpy_env:
	@conda deactivate qmcpy
	@conda remove --name qmcpy --all

gen_doc:
	# Compiling QMCPy HTML documentation.
	@$(MAKE) doc_html
	@ls -l sphinx/_build/index.html
	# Compiling QMCPy EPUB documentation.
	@$(MAKE) doc_epub
	@ls -l sphinx/_build/epub/qmcpy.epub
	# Compiling QMCPy PDF documentation.
	@$(MAKE) doc_pdf
	@ls -l sphinx/_build/qmcpy.pdf
