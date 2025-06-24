doctests_minimal:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/ \
		--ignore qmcpy/util/kernel_methods/ft_pytorch.py \
		--ignore qmcpy/stopping_criterion/pf_gp_ci.py \
		--ignore qmcpy/util/kernel_methods/dig_shift_invar_ops.py \
		--ignore qmcpy/util/exact_gpytorch_gression_model.py \
		--ignore qmcpy/integrand/umbridge_wrapper.py \
		--ignore qmcpy/integrand/hartmann6d.py

doctests_torch:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/util/kernel_methods/ft_pytorch.py \
		--doctest-modules qmcpy/util/kernel_methods/dig_shift_invar_ops.py

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
	@phmutest docs/*.md --replmode --log -c

doctests: doctests_markdown doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge

doctests_no_docker: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch

unittests:
	python -m pytest --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append test/

tests: doctests unittests coverage

tests_no_docker: doctests_no_docker unittests coverage

coverage: # https://github.com/marketplace/actions/coverage-badge
	python -m coverage report -m

mkdocserve:
	@cp CONTRIBUTING.md docs/CONTRIBUTING.md 
	@cp community.md docs/community.md 
	@mkdocs serve

uml:
	# UML Diagrams
	#	Discrete Distributions
	@pyreverse -k qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/umls/discrete_distribution_overview.svg
	#	True Measure Overview
	@pyreverse -k qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/umls/true_measure_overview.svg
	#	Integrand Overview
	@pyreverse -k qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/umls/integrand_overview.svg
	#	Stopping Criterion Overview
	@pyreverse -k qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/umls/stopping_criterion_overview.svg
	#	Discrete Distribution Specific
	@pyreverse qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/umls/discrete_distribution_specific.svg
	#	True Measure Specific
	@pyreverse qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/umls/true_measure_specific.svg
	#	Integrand Specific
	@pyreverse qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/umls/integrand_specific.svg
	#	Stopping Criterion Specific
	@pyreverse qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/umls/stopping_criterion_specific.svg
	#	Util
	@pyreverse -k qmcpy/util/ -o svg 1>/dev/null && mv classes.svg docs/umls/util_uml.svg

doc: uml mkdocserve
