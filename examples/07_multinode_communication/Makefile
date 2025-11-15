#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# This example requires MPI support for multinode communication
MPI = 1

# Include common build rules
include ../../makefiles/common.mk
include ../../makefiles/examples.mk

# Target executable
TARGET = multinode_allreduce

# Source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

# Compile source files
%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target - single node with multiple GPUs
test: $(TARGET)
	@echo "Testing $(TARGET) on single node..."
	@echo "Running with 2 processes (requires 2 GPUs)"
	$(MPIRUN) -np 2 ./$(TARGET)
	@echo ""
	@echo "Running with 4 processes (requires 4 GPUs)"
	$(MPIRUN) -np 4 ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Multinode Communication Example"
	@echo "====================================="
	@echo ""
	@echo "This example demonstrates NCCL collective operations across"
	@echo "multiple nodes with multiple GPUs per node."
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build the example (default)"
	@echo "  test          - Build and run single-node tests"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install to PREFIX/bin"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Build variables:"
	@echo "  NCCL_HOME     - Path to NCCL installation"
	@echo "  MPI_HOME      - Path to MPI installation"
	@echo "  CUDA_HOME     - Path to CUDA installation"

.PHONY: all test clean install help
