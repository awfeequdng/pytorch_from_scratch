

PYINC = $(shell python3-config --includes)
PYLIB = $(shell python3-config --ldflags)

CFLAGS = $(PYINC) -fPIC -shared -I../..
CFLAGS += $(PYLIB) -lpython3.9
# CFLAGS += -lpython3.9

Module.o: Module.cpp
	clang++ $(CFLAGS) -c $^ -o $@

utils.o: utils.cpp
	clang++ $(CFLAGS) -c $^ -o $@

all: Module.o utils.o
	clang++ -shared $(CFLAGS) $^ -o libtorch.so

clean:
	rm *.o libtorch.so