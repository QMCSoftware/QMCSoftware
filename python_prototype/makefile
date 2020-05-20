qrngpath = qmcpy/discrete_distribution/qrng/
qrng_:
	$ gcc -fPIC -shared  -o $(qrngpath)qrng_lib.so $(qrngpath)*.c -lm

test_:
	python -W ignore -m unittest discover -s test/fasttests/ 1>/dev/null
	python -W ignore -m unittest discover -s test/longtests/ 1>/dev/null

docs_:
	cd sphinx && chmod +x autodoc.sh &&  ./autodoc.sh

workouts_:
	# example_constructions
	python workouts/example_constructions/distributions.py > outputs/example_constructions/distributions.log
	python workouts/example_constructions/integrands.py > outputs/example_constructions/integrands.log
	python workouts/example_constructions/measures.py > outputs/example_constructions/measures.log
	python workouts/example_constructions/stopping_criteria.py > outputs/example_constructions/stopping_criteria.log
	# integration_examples
	python workouts/integration_examples/asian_option_multi_level.py  > outputs/integration_examples/asian_option_multi_level.log
	python workouts/integration_examples/asian_option_single_level.py  > outputs/integration_examples/asian_option_single_level.log
	##python workouts/integration_examples/free_wifi.py > outputs/integration_examples/free_wifi.log
	python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
	# lds_sequences
	python workouts/lds_sequences/python_sequences.py 
	# mc_vs_qmc
	python workouts/mc_vs_qmc/importance_sampling.py
	python workouts/mc_vs_qmc/integrations_asian_call.py
	python workouts/mc_vs_qmc/integrations_keister.py
	python workouts/mc_vs_qmc/vary_abs_tol.py
	python workouts/mc_vs_qmc/vary_dimension.py
	# mlmc
	python workouts/mlmc/mcqmc06.py > outputs/mlmc/mcqmc06.log
