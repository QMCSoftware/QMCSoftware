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
	@echo "\nFastests"
	python -W ignore -m unittest discover -s test/fasttests/ 1>/dev/null
	@echo "\nLongtests"
	python -W ignore -m unittest discover -s test/longtests/ 1>/dev/null

mddir = sphinx/readme_rst/
nbdir = sphinx/demo_rst/
nbconvertcmd = jupyter nbconvert --to rst --output-dir='$(nbdir)'
_doc:
	# Make Directries
	@-rm -r $(mddir) 2>/dev/null &
	@-rm -r $(nbdir) 2>/dev/null &
	# QMCSoftware READMEs --> RST
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
	@rm -f $(nbdir)nei_demo.rst 
	@rm -r $(nbdir)nei_demo_files/
	@-cd sphinx && make clean
doc_html: _doc
	-$(MAKE) -C sphinx html 2>/dev/null
	@-rm -r $(mddir)
	@-rm -r $(nbdir)
doc_pdf: _doc
	-$(MAKE) -C sphinx latex 2>/dev/null
	-$(MAKE) -C sphinx/_build/latex/ all-pdf -W --keep-going
	@-rm -r $(mddir)
	@-rm -r $(nbdir)
doc_epub: _doc
	-$(MAKE) -C sphinx epub 2>/dev/null
	@-rm -r $(mddir)
	@-rm -r $(nbdir)
workout:
	# integration_examples
	@python workouts/integration_examples/asian_option_multi_level.py  > outputs/integration_examples/asian_option_multi_level.log
	@python workouts/integration_examples/asian_option_single_level.py  > outputs/integration_examples/asian_option_single_level.log
	@python workouts/integration_examples/keister.py  > outputs/integration_examples/keister.log
	# lds_sequences
	@python workouts/lds_sequences/python_sequences.py 
	# mc_vs_qmc
	@python workouts/mc_vs_qmc/importance_sampling.py
	@python workouts/mc_vs_qmc/integrations_asian_call.py
	@python workouts/mc_vs_qmc/integrations_keister.py
	@python workouts/mc_vs_qmc/vary_abs_tol.py
	@python workouts/mc_vs_qmc/vary_dimension.py
	# mlmc
	@python workouts/mlmc/mcqmc06.py > outputs/mlmc/mcqmc06.log
	@python workouts/mlmc/european_option.py > outputs/mlmc/european_option.log
