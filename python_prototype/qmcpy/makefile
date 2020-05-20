cc    	= gcc
rpath = discrete_distribution/qrng/
output = qrng_lib.so
clean:
	rm -f $(rpath)$(output)
all:
	$(cc) -fPIC -shared  -o $(rpath)$(output) $(rpath)MRG63k3a.c $(rpath)ghalton.c $(rpath)korobov.c $(rpath)sobol.c -lm