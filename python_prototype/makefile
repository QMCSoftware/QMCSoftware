qrngpath = qmcpy/discrete_distribution/qrng/
qrng_:
	$ gcc -fPIC -shared  -o $(qrngpath)qrng_lib.so $(qrngpath)*.c -lm

test_:
	python -W ignore -m unittest discover -s test/fasttests/ 1>/dev/null
	python -W ignore -m unittest discover -s test/longtests/ 1>/dev/null

mddir = sphinx/readme_rst/
nbdir = sphinx/demo_rst/
nbconvertcmd = jupyter nbconvert --to rst --output-dir='$(nbdir)' demos/
docs_:
	# READMEs --> RST
	mkdir $(mddir)
	#	QMCSoftware
	echo 'Converting README: QMCSoftware'
	grep -v  "\[\!" ../README.md > ../README2.md
	pandoc --mathjax ../README2.md -o $(mddir)QMCSoftware.rst
	rm ../README2.md
	# 	python_prototype
	echo 'Converting README: python_prototype'
	pandoc --mathjax README.md -o $(mddir)python_prototype.rst
	# 	qmcpy
	echo 'Converting README: qmcpy'
	pandoc --mathjax qmcpy/README.md -o $(mddir)qmcpy.rst
	# 	test
	echo 'Converting README: test'
	pandoc --mathjax test/README.md -o $(mddir)test.rst
	# Jupyter Notebook Demos --> RST
	mkdir $(nbdir)	
	for f in demos/*.ipynb;	do
	echo "Converting Jupyter Notebook: $f"
	$(nbconvertcmd)$f
	done
	# Generate HTML
	echo 'Generating HTML'
	cd sphinx && make clean && make html
	# Generate PDF 
	echo 'Generating PDF'
	cd sphinx && make clean && make pdf
	# Generate EPUB
	echo 'Generating EPUB'
	cd sphinx && make clean && make epub

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
