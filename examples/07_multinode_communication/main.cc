/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
 * NCCL Multinode Communication Example
 * =====================================
 *
 * LEARNING OBJECTIVE:
 * This example demonstrates how to perform NCCL collective operations across
 * multiple nodes, each with multiple GPUs. This is the standard pattern for
 * large-scale distributed GPU computing.
 *
 * TARGET CONFIGURATION:
 * - 2 nodes, each with 8 GPUs
 * - Total of 16 GPUs participating in collective communication
 * - One MPI process per GPU (16 total MPI processes)
 *
 * WHAT THIS CODE DEMONSTRATES:
 * - Multi-node NCCL communicator initialization
 * - Automatic GPU assignment across multiple nodes
 * - AllReduce collective operation across all GPUs
 * - Data verification and correctness checking
 * - Best practices for multi-node NCCL applications
 *
 * STEP-BY-STEP PROCESS:
 * 1. MPI Setup: Initialize MPI across all nodes
 * 2. Node Detection: Determine which processes are on which nodes
 * 3. GPU Assignment: Map each process to a local GPU (0-7 on each node)
 * 4. NCCL Initialization: Create distributed communicator across all GPUs
 * 5. Data Preparation: Each GPU prepares test data
 * 6. Collective Operation: Perform AllReduce sum across all 16 GPUs
 * 7. Verification: Check that all GPUs received the correct result
 * 8. Cleanup: Properly destroy all resources
 */

// Enhanced error checking macro for NCCL operations
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int err = cmd;                                                             \
    if (err != MPI_SUCCESS) {                                                  \
      char errstr[MPI_MAX_ERROR_STRING];                                       \
      int len;                                                                 \
      MPI_Error_string(err, errstr, &len);                                     \
      fprintf(stderr, "Failed: MPI error %s:%d '%s'\n", __FILE__, __LINE__,    \
              errstr);                                                         \
      fprintf(stderr, "Failed MPI operation: %s\n", #cmd);                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =============================================================================
// LOCAL RANK UTILITY FUNCTION - For Multi-Node GPU Assignment
// =============================================================================

/**
 * Determine the local rank of this process on its physical node
 *
 * This function splits the MPI communicator based on shared memory regions,
 * which corresponds to physical nodes. Each process gets a local rank
 * (0, 1, 2, ..., N-1) on its node, which is used for GPU assignment.
 *
 * @param comm The MPI communicator to use for determining local rank
 * @return Local rank (0-7 for an 8-GPU node) for GPU assignment
 */
int getLocalRank(MPI_Comm comm) {
  int world_size, world_rank;
  MPICHECK(MPI_Comm_size(comm, &world_size));
  MPICHECK(MPI_Comm_rank(comm, &world_rank));

  // Split the communicator based on shared memory (i.e., nodes)
  MPI_Comm node_comm;
  MPICHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank,
                                MPI_INFO_NULL, &node_comm));

  // Get the rank within the node communicator
  int node_rank, node_size;
  MPICHECK(MPI_Comm_rank(node_comm, &node_rank));
  MPICHECK(MPI_Comm_size(node_comm, &node_size));

  // Clean up the node communicator
  MPICHECK(MPI_Comm_free(&node_comm));

  return node_rank;
}

/**
 * Get a string identifying the hostname of this node
 *
 * @param hostname Buffer to store the hostname (must be at least MPI_MAX_PROCESSOR_NAME)
 * @return 0 on success, -1 on error
 */
int getHostName(char *hostname, int maxlen) {
  int len;
  if (MPI_Get_processor_name(hostname, &len) != MPI_SUCCESS) {
    return -1;
  }
  return 0;
}

// =============================================================================
// MAIN FUNCTION - Multinode NCCL Communication Example
// =============================================================================

int main(int argc, char *argv[]) {
  // Variables for MPI, CUDA, and NCCL components
  int mpi_rank, mpi_size, local_rank;
  int num_gpus = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME];
  ncclComm_t comm = NULL;
  cudaStream_t stream = NULL;
  ncclUniqueId nccl_id;

  // Buffer pointers
  float *sendbuff = NULL;
  float *recvbuff = NULL;
  const size_t count = 32 * 1024 * 1024; // 32M floats per GPU

  // =========================================================================
  // STEP 1: Initialize MPI and determine process layout
  // =========================================================================

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  // Get hostname for debugging output
  if (getHostName(hostname, sizeof(hostname)) != 0) {
    sprintf(hostname, "unknown");
  }

  if (mpi_rank == 0) {
    printf("=================================================================\n");
    printf("NCCL Multinode Communication Example\n");
    printf("=================================================================\n");
    printf("Total MPI processes: %d\n", mpi_size);
    printf("Expected configuration: 2 nodes × 8 GPUs = 16 total GPUs\n");
    printf("-----------------------------------------------------------------\n");
  }

  // Determine which local GPU this process should use
  local_rank = getLocalRank(MPI_COMM_WORLD);

  printf("  [Rank %2d] Hostname: %-20s  Local rank: %d\n", mpi_rank, hostname,
         local_rank);

  // Wait for all processes to print their info
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 2: Setup CUDA device for this process
  // =========================================================================

  // Check how many CUDA devices are available on this node
  CUDACHECK(cudaGetDeviceCount(&num_gpus));

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("GPU Assignment:\n");
  }

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: Rank %d on %s found no CUDA devices!\n", mpi_rank,
            hostname);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  if (local_rank >= num_gpus) {
    fprintf(stderr,
            "ERROR: Rank %d on %s needs GPU %d but only %d devices available\n",
            mpi_rank, hostname, local_rank, num_gpus);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Assign this process to its designated GPU device
  CUDACHECK(cudaSetDevice(local_rank));

  // Create CUDA stream for GPU operations
  CUDACHECK(cudaStreamCreate(&stream));

  printf("  [Rank %2d] Using GPU %d on %s\n", mpi_rank, local_rank, hostname);

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 3: Initialize NCCL communicator across all nodes
  // =========================================================================

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("NCCL Initialization:\n");
  }

  // Generate NCCL unique ID (only rank 0 needs to do this)
  if (mpi_rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&nccl_id));
    printf("  [Rank  0] Generated NCCL unique ID\n");
  }

  // Share the unique ID with all processes using MPI broadcast
  MPICHECK(MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0,
                     MPI_COMM_WORLD));

  if (mpi_rank == 0) {
    printf("  [Rank  0] Broadcasting NCCL ID to all ranks...\n");
  }

  // Create NCCL communicator for this process
  // All 16 processes join the same NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));

  // Verify communicator setup
  int comm_rank, comm_size, comm_device;
  NCCLCHECK(ncclCommUserRank(comm, &comm_rank));
  NCCLCHECK(ncclCommCount(comm, &comm_size));
  NCCLCHECK(ncclCommCuDevice(comm, &comm_device));

  printf("  [Rank %2d] NCCL communicator created (NCCL rank %d/%d, GPU %d)\n",
         mpi_rank, comm_rank, comm_size, comm_device);

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 4: Allocate and initialize data buffers
  // =========================================================================

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("Data Preparation:\n");
    printf("  Buffer size per GPU: %zu floats (%.2f MB)\n", count,
           (count * sizeof(float)) / (1024.0 * 1024.0));
  }

  // Allocate device memory for send and receive buffers
  CUDACHECK(cudaMalloc((void **)&sendbuff, count * sizeof(float)));
  CUDACHECK(cudaMalloc((void **)&recvbuff, count * sizeof(float)));

  // Initialize send buffer with rank value
  // Each GPU will contribute its rank number
  float rank_value = (float)mpi_rank;
  CUDACHECK(cudaMemset(sendbuff, 0, count * sizeof(float)));
  CUDACHECK(
      cudaMemcpy(sendbuff, &rank_value, sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(recvbuff, 0, count * sizeof(float)));

  printf("  [Rank %2d] Initialized with value %.0f\n", mpi_rank, rank_value);

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 5: Perform AllReduce collective operation
  // =========================================================================

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("Performing AllReduce (Sum) across all %d GPUs...\n", mpi_size);
  }

  // Perform AllReduce Sum operation
  // All GPUs will sum their contributions and all will receive the result
  NCCLCHECK(
      ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream));

  // Synchronize stream to ensure completion
  CUDACHECK(cudaStreamSynchronize(stream));

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (mpi_rank == 0) {
    printf("  AllReduce completed successfully!\n");
  }

  // =========================================================================
  // STEP 6: Verify results
  // =========================================================================

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("Result Verification:\n");
  }

  // Expected result: sum of all ranks = 0 + 1 + 2 + ... + (mpi_size-1)
  float expected = (float)(mpi_size * (mpi_size - 1) / 2);

  // Copy result back to host for verification
  float result;
  CUDACHECK(
      cudaMemcpy(&result, recvbuff, sizeof(float), cudaMemcpyDeviceToHost));

  bool success = (result == expected);
  if (success) {
    printf("  [Rank %2d] ✓ Correct result: %.0f (expected %.0f)\n", mpi_rank,
           result, expected);
  } else {
    printf("  [Rank %2d] ✗ INCORRECT result: %.0f (expected %.0f)\n", mpi_rank,
           result, expected);
  }

  // Collect global success status
  int local_success = success ? 1 : 0;
  int global_success = 0;
  MPICHECK(MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD));

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 7: Cleanup resources
  // =========================================================================

  if (mpi_rank == 0) {
    printf("-----------------------------------------------------------------\n");
    printf("Cleanup:\n");
  }

  // Free device memory
  if (sendbuff != NULL) {
    CUDACHECK(cudaFree(sendbuff));
  }
  if (recvbuff != NULL) {
    CUDACHECK(cudaFree(recvbuff));
  }

  // Destroy CUDA stream
  if (stream != NULL) {
    CUDACHECK(cudaStreamDestroy(stream));
  }

  // Destroy NCCL communicator
  if (comm != NULL) {
    NCCLCHECK(ncclCommDestroy(comm));
  }

  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  // =========================================================================
  // STEP 8: Report final status
  // =========================================================================

  if (mpi_rank == 0) {
    printf("=================================================================\n");
    if (global_success) {
      printf("SUCCESS: All ranks verified correct results!\n");
      printf("=================================================================\n");
      printf("\nThis example demonstrated:\n");
      printf("  ✓ Multi-node NCCL communicator initialization\n");
      printf("  ✓ Automatic GPU assignment across nodes\n");
      printf("  ✓ AllReduce collective operation across %d GPUs\n", mpi_size);
      printf("  ✓ Data verification and correctness checking\n");
      printf("\nNext steps:\n");
      printf("  - Try other collectives: Broadcast, Reduce, AllGather\n");
      printf("  - Experiment with different data sizes\n");
      printf("  - Test with different numbers of nodes/GPUs\n");
      printf("  - Add timing measurements for performance analysis\n");
    } else {
      printf("FAILURE: Some ranks reported incorrect results\n");
      printf("=================================================================\n");
    }
  }

  MPICHECK(MPI_Finalize());

  return global_success ? EXIT_SUCCESS : EXIT_FAILURE;
}
