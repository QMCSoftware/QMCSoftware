doctests_minimal:
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/ \
		--ignore qmcpy/fast_transform/ft_pytorch.py \
		--ignore qmcpy/stopping_criterion/pf_gp_ci.py \
		--ignore qmcpy/kernel/ \
		--ignore qmcpy/util/dig_shift_invar_ops.py \
		--ignore qmcpy/util/shift_invar_ops.py \
		--ignore qmcpy/util/exact_gpytorch_gression_model.py \
		--ignore qmcpy/integrand/umbridge_wrapper.py \
		--ignore qmcpy/integrand/hartmann6d.py \

doctests_torch:
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/fast_transform/ft_pytorch.py \
		--doctest-modules qmcpy/kernel/*.py \
		--doctest-modules qmcpy/util/dig_shift_invar_ops.py \
		--doctest-modules qmcpy/util/shift_invar_ops.py \

doctests_gpytorch:
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/stopping_criterion/pf_gp_ci.py \

doctests_botorch:
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/hartmann6d.py \

doctests_umbridge: # https://github.com/UM-Bridge/umbridge/issues/96
	@docker --version
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/umbridge_wrapper.py \

doctests_markdown:
	@phmutest docs/*.md --replmode --log -c

doctests: doctests_markdown doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge

doctests_no_docker: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch

unittests:
	python -m pytest -x --cov qmcpy/ --cov-report term --cov-report json --no-header --cov-append test/

check_booktests:
	@echo "Checking notebook ↔ booktest coverage..."
	@rm -fr demos/.ipynb_checkpoints/*checkpoint.ipynb || true
	@total_notebooks=0; \
	skipped_notebooks=0; \
	run_notebooks=0; \
	missing_notebooks=0; \
	for nb in $$(find demos -name '*.ipynb' | sort); do \
		total_notebooks=$$((total_notebooks+1)); \
		base=$$(basename "$$nb" .ipynb); \
		test_base=$$(echo "$$base" | sed 's/[-.]/_/g'); \
		# ---------- explicit skip list ---------- \
		if [ "$$base" = "dakota_genz" ]; then \
			echo "    Skipping $$nb (requires large manual file / heavy memory use)"; \
			skipped_notebooks=$$((skipped_notebooks+1)); \
			continue; \
		fi; \
		if [ "$$base" = "parsl_fest_2025" ]; then \
			echo "    Skipping $$nb (demo orchestrates booktests, not itself a booktest)"; \
			skipped_notebooks=$$((skipped_notebooks+1)); \
			continue; \
		fi; \
		if [ "$$base" = "Argonne_2023_Talk_Figures" ]; then \
			echo "    Skipping $$nb (heavy LaTeX + many figures; skipped in booktests_no_docker)"; \
			skipped_notebooks=$$((skipped_notebooks+1)); \
			continue; \
		fi; \
		if [ "$$base" = "01_sequential" ] || [ "$$base" = "02_parallel" ] || [ "$$base" = "03_visualize_speedup" ]; then \
			echo "    Skipping $$nb (helper notebook for parsl_fest_2025; not a standalone booktest)"; \
			skipped_notebooks=$$((skipped_notebooks+1)); \
			continue; \
		fi; \
		# ---------- check for matching Testbook file ---------- \
		if ! ls test/booktests/tb_"$$test_base".py > /dev/null 2>&1; then \
			echo "    Missing test for: $$nb -> Expected: test/booktests/tb_$$test_base.py"; \
			missing_notebooks=$$((missing_notebooks+1)); \
		else \
			run_notebooks=$$((run_notebooks+1)); \
		fi; \
	done; \
	echo "Total notebooks found:        $$total_notebooks"; \
	echo "Total skipped explicitly:     $$skipped_notebooks"; \
	echo "Total notebooks WITH tests:   $$run_notebooks"; \
	echo "Total notebooks MISSING tests:$$missing_notebooks"; \
	echo "Booktests will be run for $$run_notebooks notebooks."


# This helps locate generated or local-only folders like build, .pytest_cache, etc.
find_local_only_files:
	@chmod +x scripts/find_local_only_folders.sh
	@./scripts/find_local_only_folders.sh

clean_local_only_files:
	@rm -fr test/booktests/.ipynb_checkpoints/
	@chmod +x scripts/find_local_only_folders.sh
	@./scripts/find_local_only_folders.sh | while read f; do \
		if [ -n "$$f" ]; then \
			rm -f "$$f" > /dev/null 2>&1; \
		fi; \
	done

generate_booktests:
	@echo ""
	@echo "Generating missing booktest files..."
	@cd test/booktests/ && python generate_test.py --check-missing


booktests_no_docker: check_booktests generate_booktests clean_local_only_files
	@echo "\nNotebook tests"
	@pip install -q -e ".[test]"  && \
	set -e && \
	cd test/booktests/ && \
	if [ -z "$(TESTS)" ]; then \
		TOTAL_NOTEBOOK_TESTS=$$(find . -maxdepth 1 -name 'tb_*.py' | wc -l) \
		PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
		TOTAL_NOTEBOOK_TESTS=$$TOTAL_NOTEBOOK_TESTS python -W ignore -m coverage run --append --source=../../qmcpy/ -m unittest discover -s . -p "*.py" -v --failfast; \
	else \
		TOTAL_NOTEBOOK_TESTS=$$(echo "$(TESTS)" | wc -w) \
		PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
		TOTAL_NOTEBOOK_TESTS=$$TOTAL_NOTEBOOK_TESTS python -W ignore -m coverage run --append --source=../../qmcpy/ -m unittest $(TESTS) -v --failfast; \
	fi && \
	cd ../..

booktests_parallel_no_docker: check_booktests generate_booktests clean_local_only_files
	@echo "\nNotebook tests with Parsl"
	@pip install -q -e ".[test]"  && \
	cd test/booktests/ && \
	rm -fr *.eps *.jpg *.pdf *.png *.part *.txt *.log && rm -fr logs && rm -fr runinfo prob_failure_gp_ci_plots && \
	PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
	python parsl_test_runner.py $(TESTS) -v --failfast && \
	cd ../..

tests: 
	set -e && $(MAKE) doctests && $(MAKE) unittests && $(MAKE) coverage

tests_no_docker: 
	pip install -q -e .[test] && \
	set -e && $(MAKE) doctests_no_docker && $(MAKE) unittests && $(MAKE) coverage

coverage: # https://github.com/marketplace/actions/coverage-badge
	python -m coverage report -m

delcoverage:
	@rm .coverage
	@rm coverage.json 

uml:
	# UML Diagrams
	#	Discrete Distributions
	@pyreverse -k qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/discrete_distribution_overview.svg
	#	Discrete Distribution Specific
	@pyreverse qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/discrete_distribution_specific.svg
	#	True Measure Overview
	@pyreverse -k qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/true_measure_overview.svg
	#	True Measure Specific
	@pyreverse qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/true_measure_specific.svg
	#	Integrand Overview
	@pyreverse -k qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/integrand_overview.svg
	#	Integrand Specific
	@pyreverse qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/integrand_specific.svg
	#	Stopping Criterion Overview
	@pyreverse -k qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/stopping_criterion_overview.svg
	#	Stopping Criterion Specific
	@pyreverse qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/stopping_criterion_specific.svg
	#	Kernel Overview
	@pyreverse -k qmcpy/kernel/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/kernel_overview.svg
	#	Kernel Specific
	@pyreverse qmcpy/kernel/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/kernel_specific.svg

copydocs: # mkdocs only looks for content in the docs/ folder, so we have to copy it there
	@cp README.md docs/README.md 
	@cp CONTRIBUTING.md docs/CONTRIBUTING.md 
	@cp community.md docs/community.md 
	@cp -r demos docs

runmkdocserve: 
	@mkdocs serve
	
doc: uml copydocs runmkdocserve

docnouml: copydocs runmkdocserve