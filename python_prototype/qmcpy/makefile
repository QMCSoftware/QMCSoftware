<<<<<<< HEAD
# Makefile for compiling dynamic QRNGPY library for various platforms
#
# Type at commpand prompt the following line to compile qrngpy dynamic library
#    make all
#
# Type at command prompt the following line to remove dynamic library
#    make clean
#

# define OS
UNAME := $(shell uname)

# define extension name of shared library
ifeq ($(UNAME), Linux)
EXT = so
endif
ifeq ($(UNAME), Darwin)
EXT = dylib
endif
ifeq ($(UNAME), Windows)
EXT = dll
endif

# define C compiler
CC    	= gcc

# define QRNGPY relative path
rpath = discrete_distribution/qrngpy/

# define target for removing dynamic library
clean:
	rm -f $(rpath)*.o $(rpath)*.$(EXT)

# define target for compiling dynamic library
all:
	$(CC) -fPIC -shared  -o $(rpath)qrngpy.$(EXT) $(rpath)MRG63k3a.c $(rpath)ghalton.c $(rpath)korobov.c $(rpath)sobol.c -lm





=======
cc    	= gcc
rpath = discrete_distribution/qrng/
output = qrng_lib.so
clean:
	rm -f $(rpath)$(output)
all:
	$(cc) -fPIC -shared  -o $(rpath)$(output) $(rpath)MRG63k3a.c $(rpath)ghalton.c $(rpath)korobov.c $(rpath)sobol.c -lm
>>>>>>> update fasttests
