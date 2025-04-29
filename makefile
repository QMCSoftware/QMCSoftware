doctests_minimal:
	python -m coverage run --source=./qmcpy/ -m pytest --doctest-modules --disable-pytest-warnings qmcpy --no-header --ignore qmcpy/integrand/um_bridge_wrapper.py --ignore qmcpy/accumulate_data/pf_gp_ci_data.py --ignore qmcpy/stopping_criterion/pf_gp_ci.py

doctests_no_docker:
	@echo "\nDoctests Without Docker Containers"
	python -m coverage run --source=./qmcpy/ -m pytest --doctest-modules --ignore qmcpy/integrand/um_bridge_wrapper.py --disable-pytest-warnings qmcpy

fasttests:
	@echo "\nFastests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/fasttests/ 1>/dev/null

longtests:
	@echo "\nLongtests"
	python -W ignore -m coverage run --append --source=./ -m unittest discover -s test/longtests/ 1>/dev/null

coverage:
	@echo "\nCode coverage"
	python -m coverage report -m

tests: doctests fasttests longtests coverage

tests_no_docker: doctests_no_docker fasttests longtests coverage

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
	
mkdocs_serve:
	@cp README.md docs/index.md
	@mkdocs serve

uml:
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

doc: uml mkdocserve

export_conda_env:
	@-rm -f environment.yml 2>/dev/null &
	@conda env export --no-builds | grep -v "^prefix: " > environment.yml
