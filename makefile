doctests_minimal:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/ \
		--ignore qmcpy/util/kernel_methods/ft_pytorch.py \
		--ignore qmcpy/accumulate_data/pf_gp_ci_data.py \
		--ignore qmcpy/stopping_criterion/pf_gp_ci.py \
		--ignore qmcpy/util/exact_gpytorch_gression_model.py \
		--ignore qmcpy/integrand/umbridge_wrapper.py \
		--ignore qmcpy/integrand/hartmann6d.py

doctests_torch:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/util/kernel_methods/ft_pytorch.py

doctests_gpytorch:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/stopping_criterion/pf_gp_ci.py

doctests_botorch:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/hartmann6d.py

doctests_umbridge: # https://github.com/UM-Bridge/umbridge/issues/96
	@docker --version
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/umbridge_wrapper.py

doctests_markdown:
	@phmutest docs/discrete_distributions.md --replmode --log

doctests: doctests_readme doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge

doctests_no_docker: doctests_readme doctests_minimal doctests_torch doctests_gpytorch doctests_botorch

unittests:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append test/

coverage: # https://github.com/marketplace/actions/coverage-badge
	python -m coverage report -m

tests: doctests unittests coverage

tests_no_docker: doctests_no_docker unittests coverage
	
mkdocs_serve:
	@cp CONTRIBUTING.md docs/CONTRIBUTING.md 
	@cp community.md docs/community.md 
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
