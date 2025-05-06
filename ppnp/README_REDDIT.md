# Reddit Dataset Extension for PPNP

This extension adds support for the Reddit dataset to the PPNP codebase, optimized for running on systems with limited GPU memory (2 GPUs with 10GB each).

## Overview

The Reddit dataset is significantly larger than the Planetoid datasets (Cora, Citeseer, PubMed) originally supported by this codebase. The Reddit dataset contains:
- 232,965 nodes (Reddit posts)
- 114,615,892 edges (post-to-post connections)
- 602 input features per node
- 41 classes (subreddit categories)

Running graph neural networks on such a large dataset requires memory optimization techniques and distributed processing.

## Requirements

- 2 CUDA-compatible GPUs with at least 10GB memory each
- PyTorch with CUDA support
- PyTorch Geometric with relevant extensions

## How to Run

We've provided a specialized script `run_reddit.sh` that configures the appropriate parameters for the Reddit dataset:

```bash
# Basic usage (with default parameters)
bash run_reddit.sh

# With custom batch size and neighbor sampling
bash run_reddit.sh --batch_size 256 --num_neighbors 5,5

# To run on a single GPU (not recommended due to memory constraints)
bash run_reddit.sh --no_parallel
```

### Command-line Options

- `--batch_size SIZE`: Set mini-batch size (default: 256)
- `--num_neighbors N1,N2`: Set number of neighbors to sample at each hop (default: 5,5)
- `--epochs N`: Set number of training epochs (default: 30)
- `--hidden DIM`: Set hidden dimension size (default: 32)
- `--k_steps K`: Set number of propagation steps (default: 5)
- `--alpha ALPHA`: Set teleport probability (default: 0.1)
- `--world_size N`: Set number of GPUs to use (default: 2)
- `--no_parallel`: Run without parallelization (single GPU mode)
- `--debug`: Enable detailed debug logging

## Memory Optimizations

The codebase includes several memory optimizations for the Reddit dataset:

1. **Feature Dimensionality Reduction**: The original 602 features are reduced to 128 dimensions using a memory-efficient PCA implementation.
2. **Mini-batch Training**: Instead of full-graph training, we use neighbor sampling with configurable batch sizes.
3. **Distributed Processing**: The computation is split across multiple GPUs using PyTorch's distributed training capabilities.
4. **Mixed Precision Training**: We use half-precision (FP16) for certain operations to reduce memory usage.
5. **Memory Management**: Careful garbage collection and CUDA cache clearing to avoid memory leaks.
6. **Safe Graph Partitioning**: Special partitioning strategy for large graphs to prevent index out-of-bounds errors.
7. **Numerical Stability**: NaN/Inf detection and handling throughout the computation pipeline.
8. **Error Recovery**: Fallback mechanisms that gracefully handle potential errors.

## Performance Considerations

- **Batch Size**: Larger batch sizes improve training speed but consume more memory. The default of 256 is a balanced choice for 10GB GPUs.
- **Neighbor Sampling**: The `num_neighbors` parameter controls how many neighbors to sample at each hop. Lower values use less memory but may reduce model accuracy.
- **Hidden Dimension**: Reducing the hidden dimension (`--hidden`) can significantly decrease memory usage but may affect model capacity.
- **Propagation Steps**: The `k_steps` parameter controls the number of propagation steps in the APPNP model. Lower values use less memory but may reduce message passing effectiveness.

## Recent Fixes

This extension includes several critical fixes for running on the Reddit dataset:

1. **Index Out-of-Bounds Fix**: The parallel BSP implementation now properly handles large graph indices and validates all tensor accesses.
2. **Memory-Efficient Partitioning**: For Reddit's large graph structure, we've implemented balanced partitioning that avoids METIS overhead.
3. **Robust Edge Handling**: Special handling for edge dropout in large graphs to prevent numerical instability.
4. **Checkpointing Improvements**: Enhanced gradient checkpointing for reduced memory footprint.
5. **Automatic Parameter Adjustment**: The model automatically adjusts its parameters when working with large graphs like Reddit.

## Monitoring GPU Memory

To monitor GPU memory usage during training:

```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

## Troubleshooting

If you encounter CUDA out-of-memory errors:

1. Reduce the batch size (`--batch_size 128` or even smaller)
2. Reduce neighbor sampling (`--num_neighbors 3,3` or even `2,2`)
3. Reduce hidden dimension (`--hidden 16`)
4. Reduce propagation steps (`--k_steps 3`)
5. Try running with gradient checkpointing (enabled by default)
6. Ensure you're using both GPUs (`--world_size 2`)

### Specific Error Solutions

#### CUDA Error: device-side assert triggered

If you encounter "CUDA error: device-side assert triggered" or "index out of bounds" errors:

1. The script now automatically uses a safer partitioning strategy for Reddit
2. Set `--k_steps 3` to reduce the number of propagation steps
3. Try setting the environment variable `export CUDA_LAUNCH_BLOCKING=1` to get more informative error messages
4. Use the script's fallback mode which will automatically try with safer settings if the initial run fails

#### NaN/Inf Values

If you encounter NaN or Inf values in training:
1. The code now automatically detects and replaces NaN/Inf values
2. Ensure your dataset is properly normalized
3. Consider reducing the learning rate

## Results

The extended codebase should achieve comparable accuracy to the original implementation while enabling training on the much larger Reddit dataset with limited GPU resources. 