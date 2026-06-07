# DAFIM MPI build
#
# Example:
#   make
#   mpirun -np 8 ./bin/dafim_mpi --edge <edge.txt> --attr <attr.txt> \
#       --metis-parts 32 --p-default 0.01 --ic-trials 500 --scheme inv --lambda-homo 2.0
#
# If METIS is not in a standard location:
#   make METIS_INC=/path/to/metis/include METIS_LIB=/path/to/metis/lib

# IMPORTANT:
#   We intentionally build with the MPI compiler wrapper (mpicxx/mpic++),
#   because it provides the correct include paths for <mpi.h>.
#
#   If your environment sets CXX=clang++ (common on macOS), you'll see:
#       fatal error: 'mpi.h' file not found
#   In that case either:
#       make MPICXX=mpicxx
#   or install OpenMPI/MPICH and ensure mpicxx is on PATH.

MPICXX ?= mpicxx
CXX := $(MPICXX)
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter

METIS_INC ?=
METIS_LIB ?=

BIN_DIR := bin
SRC_DIR := src
OBJ_DIR := build

SOURCES := $(SRC_DIR)/main.cpp $(SRC_DIR)/dafim_core.cpp
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))
TARGET := $(BIN_DIR)/dafim_mpi

INCLUDES := -Iinclude
METIS_INCFLAGS := $(if $(METIS_INC),-I$(METIS_INC),)
METIS_LDFLAGS := $(if $(METIS_LIB),-L$(METIS_LIB),)
LIBS := -lmetis -lm

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ $(METIS_LDFLAGS) $(LIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(METIS_INCFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)
