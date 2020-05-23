# Makefile for compiling dynamic QRNGPY library for various platforms

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
# do something Linux-y
endif
ifeq ($(UNAME), Solaris)
# do something Solaris-y
endif

# Mac OS
ifeq ($(UNAME), Darwin)
all:
	gcc MRG63k3a.c -o MRG63k3a.o -c
	gcc ghalton.c -o ghalton.o -c
	gcc korobov.c -o korobov.o -c
	gcc  sobol.c -o  sobol.o -c

	g++ -dynamiclib -fPIC -o qrngpy.dylib *.o

clean:
	rm *.o
	rm *.dylib
endif


