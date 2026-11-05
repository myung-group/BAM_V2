# Build the RACEv2 PJRT runtime test on a GPU login node (aarch64, same as the GPU compute nodes):
#   module load gcc-native/14 && make -j4
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fPIC
all: test_pjrt
racev2_pjrt.o: racev2_pjrt.cc racev2_pjrt.h third_party/pjrt_c_api.h
	$(CXX) $(CXXFLAGS) -I. -c racev2_pjrt.cc -o $@
test_pjrt: test_pjrt.cc racev2_pjrt.o
	$(CXX) $(CXXFLAGS) -I. test_pjrt.cc racev2_pjrt.o -ldl -o $@
clean:
	rm -f *.o test_pjrt
