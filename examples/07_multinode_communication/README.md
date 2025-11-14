<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Multinode Communication Example

This example demonstrates how to perform NCCL collective operations across multiple nodes, each with multiple GPUs. This is the standard pattern for large-scale distributed GPU computing used in modern deep learning training and HPC applications.

## Overview

This example shows a complete multinode NCCL application that performs an AllReduce operation across multiple nodes. The example is designed for a configuration of **2 nodes with 8 GPUs each (16 total GPUs)**, but can be adapted for different configurations.

### What This Example Does

1. **Multi-node Setup**: Initializes MPI across multiple nodes and automatically detects the node topology
2. **GPU Assignment**: Intelligently maps MPI processes to local GPUs on each node (ranks 0-7 on node 1, ranks 8-15 on node 2)
3. **NCCL Initialization**: Creates a single NCCL communicator spanning all GPUs across all nodes
4. **Data Preparation**: Each GPU prepares test data (initialized with its rank value)
5. **AllReduce Operation**: Performs a sum reduction across all 16 GPUs, with all GPUs receiving the result
6. **Verification**: Validates that all GPUs received the correct summed result
7. **Cleanup**: Properly destroys all resources in the correct order

### Key Learning Points

- How to set up NCCL for multinode communication
- Automatic per-node GPU assignment using MPI
- NCCL communicator initialization across nodes
- Performing collective operations across nodes
- Verification and debugging techniques
- Best practices for cleanup and error handling

## Prerequisites

### Hardware Requirements

- **Minimum**: 2 nodes, each with at least 1 GPU
- **Recommended**: 2 nodes, each with 8 GPUs
- **Network**: High-speed interconnect between nodes (InfiniBand, RoCE, or high-speed Ethernet)

### Software Requirements

- NCCL library (2.0 or later)
- CUDA Toolkit (compatible with your GPUs)
- MPI implementation (OpenMPI, MPICH, or other)
- C++ compiler with C++11 support

### Network Configuration

For multi-node NCCL to work properly:
- Nodes must be able to communicate over the network
- Firewall rules must allow traffic between nodes
- Network interfaces must be properly configured
- For best performance, use RDMA-capable networks (InfiniBand or RoCE)

## Building the Example

### Option 1: Build from NCCL Source Directory

If you're in the main NCCL source tree:

```bash
cd nccl
make -j examples MPI=1
```

This will build all examples including the multinode example.

### Option 2: Build the Example Directly

From the example directory:

```bash
cd examples/07_multinode_communication
make
```

### Option 3: Build with Custom Paths

If NCCL, MPI, or CUDA are installed in non-standard locations:

```bash
make NCCL_HOME=/path/to/nccl \
     MPI_HOME=/path/to/mpi \
     CUDA_HOME=/path/to/cuda
```

### Build Variables

- `NCCL_HOME`: Path to NCCL installation (default: system path)
- `MPI_HOME`: Path to MPI installation (default: system path)
- `CUDA_HOME`: Path to CUDA installation (default: /usr/local/cuda)

## Running the Example

### Single Node Testing (Development/Testing)

Before running on multiple nodes, test on a single node:

```bash
# Test with 2 GPUs on one node
mpirun -np 2 ./multinode_allreduce

# Test with 4 GPUs on one node
mpirun -np 4 ./multinode_allreduce

# Test with 8 GPUs on one node
mpirun -np 8 ./multinode_allreduce
```

### Multi-Node Execution (Production Configuration)

#### Two Nodes with 8 GPUs Each (16 Total)

This is the target configuration for this example:

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       ./multinode_allreduce
```

Or with a hostfile:

```bash
# Create hostfile
cat > hostfile << EOF
node1 slots=8
node2 slots=8
EOF

# Run with hostfile
mpirun -np 16 --hostfile hostfile ./multinode_allreduce
```

#### Using Specific Network Interfaces

If you have multiple network interfaces, specify which one NCCL should use:

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       -x NCCL_SOCKET_IFNAME=eth0 \
       ./multinode_allreduce
```

For InfiniBand:

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       -x NCCL_SOCKET_IFNAME=ib0 \
       -x NCCL_IB_DISABLE=0 \
       ./multinode_allreduce
```

#### With NCCL Debug Output

To see detailed NCCL communication information:

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       -x NCCL_DEBUG=INFO \
       ./multinode_allreduce
```

For even more detailed debugging:

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       -x NCCL_DEBUG=TRACE \
       -x NCCL_DEBUG_SUBSYS=ALL \
       ./multinode_allreduce
```

### Different MPI Implementations

#### OpenMPI

```bash
mpirun -np 16 \
       --host node1:8,node2:8 \
       -x NCCL_SOCKET_IFNAME=eth0 \
       ./multinode_allreduce
```

#### MPICH/Intel MPI

```bash
mpirun -np 16 \
       -hosts node1,node2 \
       -ppn 8 \
       -env NCCL_SOCKET_IFNAME eth0 \
       ./multinode_allreduce
```

#### SLURM with srun

```bash
srun -N 2 \
     --ntasks-per-node=8 \
     --gres=gpu:8 \
     ./multinode_allreduce
```

## Expected Output

### Successful Run (2 Nodes, 16 GPUs)

```
=================================================================
NCCL Multinode Communication Example
=================================================================
Total MPI processes: 16
Expected configuration: 2 nodes × 8 GPUs = 16 total GPUs
-----------------------------------------------------------------
  [Rank  0] Hostname: node1                 Local rank: 0
  [Rank  1] Hostname: node1                 Local rank: 1
  [Rank  2] Hostname: node1                 Local rank: 2
  [Rank  3] Hostname: node1                 Local rank: 3
  [Rank  4] Hostname: node1                 Local rank: 4
  [Rank  5] Hostname: node1                 Local rank: 5
  [Rank  6] Hostname: node1                 Local rank: 6
  [Rank  7] Hostname: node1                 Local rank: 7
  [Rank  8] Hostname: node2                 Local rank: 0
  [Rank  9] Hostname: node2                 Local rank: 1
  [Rank 10] Hostname: node2                 Local rank: 2
  [Rank 11] Hostname: node2                 Local rank: 3
  [Rank 12] Hostname: node2                 Local rank: 4
  [Rank 13] Hostname: node2                 Local rank: 5
  [Rank 14] Hostname: node2                 Local rank: 6
  [Rank 15] Hostname: node2                 Local rank: 7
-----------------------------------------------------------------
GPU Assignment:
  [Rank  0] Using GPU 0 on node1
  [Rank  1] Using GPU 1 on node1
  [Rank  2] Using GPU 2 on node1
  [Rank  3] Using GPU 3 on node1
  [Rank  4] Using GPU 4 on node1
  [Rank  5] Using GPU 5 on node1
  [Rank  6] Using GPU 6 on node1
  [Rank  7] Using GPU 7 on node1
  [Rank  8] Using GPU 0 on node2
  [Rank  9] Using GPU 1 on node2
  [Rank 10] Using GPU 2 on node2
  [Rank 11] Using GPU 3 on node2
  [Rank 12] Using GPU 4 on node2
  [Rank 13] Using GPU 5 on node2
  [Rank 14] Using GPU 6 on node2
  [Rank 15] Using GPU 7 on node2
-----------------------------------------------------------------
NCCL Initialization:
  [Rank  0] Generated NCCL unique ID
  [Rank  0] Broadcasting NCCL ID to all ranks...
  [Rank  0] NCCL communicator created (NCCL rank 0/16, GPU 0)
  [Rank  1] NCCL communicator created (NCCL rank 1/16, GPU 1)
  [... output for ranks 2-15 ...]
-----------------------------------------------------------------
Data Preparation:
  Buffer size per GPU: 33554432 floats (128.00 MB)
  [Rank  0] Initialized with value 0
  [Rank  1] Initialized with value 1
  [... output for ranks 2-15 ...]
-----------------------------------------------------------------
Performing AllReduce (Sum) across all 16 GPUs...
  AllReduce completed successfully!
-----------------------------------------------------------------
Result Verification:
  [Rank  0] ✓ Correct result: 120 (expected 120)
  [Rank  1] ✓ Correct result: 120 (expected 120)
  [... output for ranks 2-15 ...]
-----------------------------------------------------------------
Cleanup:
=================================================================
SUCCESS: All ranks verified correct results!
=================================================================

This example demonstrated:
  ✓ Multi-node NCCL communicator initialization
  ✓ Automatic GPU assignment across nodes
  ✓ AllReduce collective operation across 16 GPUs
  ✓ Data verification and correctness checking

Next steps:
  - Try other collectives: Broadcast, Reduce, AllGather
  - Experiment with different data sizes
  - Test with different numbers of nodes/GPUs
  - Add timing measurements for performance analysis
```

## Code Walkthrough

### Multi-Node GPU Assignment

The example uses MPI's `MPI_Comm_split_type` to automatically determine which processes are on the same node:

```c
int getLocalRank(MPI_Comm comm) {
  // Split communicator based on shared memory (i.e., physical nodes)
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank,
                      MPI_INFO_NULL, &node_comm);

  // Get rank within this node (0, 1, 2, ..., 7 for 8 GPUs)
  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);

  MPI_Comm_free(&node_comm);
  return node_rank;
}
```

This ensures that:
- Ranks 0-7 get GPUs 0-7 on node 1
- Ranks 8-15 get GPUs 0-7 on node 2

### NCCL Communicator Initialization

All processes join a single NCCL communicator:

```c
// Rank 0 generates the unique ID
if (mpi_rank == 0) {
  ncclGetUniqueId(&nccl_id);
}

// Broadcast to all processes across all nodes
MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

// All processes join the communicator
ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank);
```

### AllReduce Operation

Each GPU performs the AllReduce with its local buffers:

```c
// All ranks call ncclAllReduce
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

// Synchronize to ensure completion
cudaStreamSynchronize(stream);
```

## Performance Considerations

### Network Bandwidth

- **InfiniBand/RoCE**: Best performance for multinode NCCL
- **High-speed Ethernet**: Good performance with proper tuning
- **Standard Ethernet**: Will work but may be bottlenecked

### Data Size

The example uses 32M floats (128 MB) per GPU. For performance testing:
- Larger buffers: Better bandwidth utilization
- Smaller buffers: Lower latency, but may not saturate network

### GPU Topology

NCCL automatically detects and optimizes for:
- NVLink connections between GPUs
- PCIe topology
- Network topology between nodes

## Common Issues and Solutions

### Issue 1: "Connection refused" or "No route to host"

**Cause**: Network connectivity issues between nodes

**Solutions**:
- Verify nodes can ping each other
- Check firewall rules (may need to open ports)
- Ensure both nodes are on the same network
- Try: `ssh node2 hostname` from node1 to verify connectivity

### Issue 2: "No CUDA devices found"

**Cause**: GPUs not visible to processes

**Solutions**:
- Verify GPUs are available: `nvidia-smi`
- Check CUDA installation: `nvidia-smi`
- Ensure CUDA driver is loaded
- Check that processes have permission to access GPUs

### Issue 3: "Failed to bind to device"

**Cause**: Too many processes for available GPUs

**Solutions**:
- Ensure: `processes_per_node ≤ GPUs_per_node`
- For 8 GPUs per node, use at most 8 processes per node
- Check with: `nvidia-smi` to see how many GPUs are available

### Issue 4: NCCL initialization hangs

**Cause**: Communication blocked between nodes

**Solutions**:
- Set network interface explicitly:
  ```bash
  -x NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
  ```
- Enable debug output to see where it hangs:
  ```bash
  -x NCCL_DEBUG=INFO
  ```
- Check if nodes can reach each other on the specified interface

### Issue 5: Poor performance or slow communication

**Cause**: Not using optimal network path

**Solutions**:
- For InfiniBand, ensure NCCL IB support is enabled:
  ```bash
  -x NCCL_IB_DISABLE=0
  ```
- Check network interface is correct:
  ```bash
  -x NCCL_DEBUG=INFO  # Look for "Using network" messages
  ```
- Verify network bandwidth:
  ```bash
  iperf3 -s  # On node1
  iperf3 -c node1  # On node2
  ```

### Issue 6: "MPI_Comm_split_type not found"

**Cause**: Old MPI version that doesn't support MPI_COMM_TYPE_SHARED

**Solutions**:
- Upgrade to MPI 3.0 or later
- Or use environment variables for GPU assignment:
  ```c
  // Fallback method
  int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  ```

## Environment Variables

### NCCL Environment Variables

- `NCCL_DEBUG`: Set to `INFO` or `TRACE` for debug output
- `NCCL_DEBUG_SUBSYS`: Set to `ALL`, `INIT`, `COLL`, `P2P`, `NET`, etc.
- `NCCL_SOCKET_IFNAME`: Network interface to use (e.g., `eth0`, `ib0`)
- `NCCL_IB_DISABLE`: Set to `0` to enable InfiniBand (default is auto-detect)
- `NCCL_NET_GDR_LEVEL`: GPUDirect RDMA level (0=disabled, 1-5=enabled)
- `NCCL_P2P_DISABLE`: Set to `1` to disable P2P between GPUs
- `NCCL_SHM_DISABLE`: Set to `1` to disable shared memory transport

### MPI Environment Variables

- `OMPI_COMM_WORLD_LOCAL_RANK`: OpenMPI local rank (set automatically)
- `MV2_COMM_WORLD_LOCAL_RANK`: MVAPICH2 local rank (set automatically)

## Advanced Usage

### Different Numbers of Nodes/GPUs

The example automatically adapts to different configurations:

```bash
# 4 nodes with 4 GPUs each (16 total)
mpirun -np 16 --host node1:4,node2:4,node3:4,node4:4 ./multinode_allreduce

# 2 nodes with 4 GPUs each (8 total)
mpirun -np 8 --host node1:4,node2:4 ./multinode_allreduce

# 8 nodes with 8 GPUs each (64 total)
mpirun -np 64 --host node[1-8]:8 ./multinode_allreduce
```

### Adding Timing Measurements

To measure performance, add timing around the AllReduce:

```c
// Start timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, stream);

// Perform AllReduce
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);

// Stop timing
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

// Calculate bandwidth
float bytes = count * sizeof(float) * 2; // Send and receive
float bandwidth_gbps = (bytes / milliseconds / 1e6);
printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
```

### Other Collective Operations

The same pattern works for other collectives:

```c
// Broadcast from rank 0 to all
ncclBroadcast(sendbuff, recvbuff, count, ncclFloat, 0, comm, stream);

// Reduce to rank 0
ncclReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, 0, comm, stream);

// AllGather
ncclAllGather(sendbuff, recvbuff, count, ncclFloat, comm, stream);

// ReduceScatter
ncclReduceScatter(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
```

## Next Steps

After understanding this example:

1. **Experiment with different collectives**: Try Broadcast, Reduce, AllGather
2. **Add performance measurements**: Measure bandwidth and latency
3. **Test with different data sizes**: See how performance scales
4. **Try different network configurations**: Test with different interfaces
5. **Scale to more nodes**: Test with 4, 8, or more nodes
6. **Integrate into applications**: Use this pattern in your distributed training code

## Related Examples

- `01_communicators/03_one_device_per_process_mpi/`: Basic MPI communicator setup
- `03_collectives/01_allreduce/`: Single-node AllReduce example
- `02_point_to_point/`: Point-to-point communication examples

## Additional Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL Tests Repository](https://github.com/NVIDIA/nccl-tests/)
- [MPI Documentation](https://www.mpi-forum.org/docs/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
