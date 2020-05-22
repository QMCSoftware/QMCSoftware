CC    	= gcc
rpath = discrete_distribution/qrngpy/

clean:
	rm -f $(TARGET)
all:
	$(CC) -fPIC -shared  -o $(rpath)qrngpy.so $(rpath)MRG63k3a.c $(rpath)ghalton.c $(rpath)korobov.c $(rpath)sobol.c -lm