# Documenting Speedups with Parallel Implementation

This guide explains how to benchmark and document the performance improvements achieved by the parallel implementation of the graph-based recommendation system.

## Quick Comparison

For a quick comparison between sequential and parallel implementations:

```bash
# Make the script executable
chmod +x run_comparison.sh

# Run with default settings (Cora dataset, 50 epochs, world_size=2)
./run_comparison.sh

# Example: Run benchmark with world size 2 (requires 2 GPUs)
./run_comparison.sh --dataset Citeseer --epochs 100 --world_size 2

# Note: Using world_size 4 requires 4 GPUs 
# Only use if you have sufficient hardware
# ./run_comparison.sh --dataset Citeseer --epochs 100 --world_size 4
```

This will:
1. Run the sequential implementation
2. Run the parallel implementation with specified world size
3. Generate a speedup comparison report and visualization

## Comprehensive Benchmarking

For a more thorough benchmark across multiple configurations:

```bash
# Run benchmarks with default settings
python benchmark_parallel.py

# Customize the benchmark parameters
python benchmark_parallel.py --dataset Cora --world_sizes 2 4 8 --epochs 50 --repeat 3

# Only analyze existing benchmark results (no new runs)
python benchmark_parallel.py --analyze_only

# Generate a comprehensive report
python benchmark_parallel.py --report

# Basic benchmark on Cora dataset with world sizes 1 and 2
# Note: Using larger world sizes (4, 8) requires corresponding number of GPUs
python benchmark_parallel.py --dataset Cora --world_sizes 1 2 --epochs 50 --repeat 3
```

## Manual Analysis

You can also use the analysis functions directly in Python:

```python
from profiling import compare_performance, strong_scaling_analysis

# Compare sequential vs parallel performance
compare_performance(
    parallel_log_file="appnp_training_log.csv",
    sequential_log_file="appnp_training_log_not_parallel.csv",
    output_file="my_analysis.json",
    plot_file="my_speedup_plot.png"
)

# Analyze scaling efficiency across different world sizes
strong_scaling_analysis(
    log_files={
        1: "sequential_log.csv",
        2: "parallel_world2_log.csv",
        4: "parallel_world4_log.csv",
        8: "parallel_world8_log.csv"
    },
    output_file="scaling_analysis.json",
    plot_file="scaling_plot.png"
)
```

## Understanding the Metrics

The analysis tools measure several performance aspects:

### Time-based Metrics
- **Overall Speedup**: Ratio of sequential epoch time to parallel epoch time
- **Training Speedup**: Speedup specific to model training phase
- **Evaluation Speedup**: Speedup specific to model evaluation phase
- **Forward/Backward Pass Speedup**: Detailed timing for specific components

### Resource Utilization
- **Memory Usage**: Comparison of GPU and CPU memory consumption
- **Memory Efficiency**: How efficiently memory is used in parallel vs sequential

### Scaling Efficiency
- **Strong Scaling**: How speedup changes as you add more processes
- **Parallel Efficiency**: Speedup divided by number of processes (ideal = 1.0)
- **Amdahl's Law Analysis**: Estimation of serial fraction and maximum theoretical speedup

### Model Accuracy
- **Accuracy Impact**: Whether parallelization affects model accuracy
- **Convergence Rate**: Number of epochs needed to reach best validation accuracy

## Example Report Structure

The generated reports include:

1. **Overview**: Summary of parallel implementation and testing environment
2. **Speedup Summary**: Breakdown of speedup across different metrics
3. **Performance Metrics**: Detailed comparison tables
4. **Scaling Analysis**: Charts showing how performance scales with world size
5. **Accuracy Impact**: Any differences in model accuracy
6. **Conclusion**: Key findings and recommendations

## Best Practices for Documentation

When documenting speedups for your project:

1. **Consistency**: Use the same hardware, dataset, and hyperparameters for valid comparisons
2. **Multiple Runs**: Average over multiple runs to account for system variability
3. **Warmup Epochs**: Exclude the first few epochs from analysis to avoid initialization costs
4. **Scaling Tests**: Test with different world sizes to find the optimal configuration
5. **Accuracy Validation**: Ensure that parallelization doesn't sacrifice model quality
6. **Resource Measurements**: Include memory and CPU/GPU utilization in reports
7. **Visualizations**: Use charts to clearly illustrate performance gains

## Interpreting Results

- **Linear Speedup**: A perfect scenario where doubling processes halves runtime
- **Sub-linear Speedup**: Common in practice due to communication overhead and serial portions
- **Diminishing Returns**: The point where adding more processes gives minimal benefit
- **Optimal Configuration**: The world size that balances speedup with resource efficiency

## Troubleshooting

If you encounter performance issues:

- **Communication Bottlenecks**: Check for excessive data transfer between processes
- **Load Imbalance**: Ensure graph partitioning is distributing work evenly
- **Memory Issues**: Monitor for memory leaks or excessive GPU memory usage
- **Synchronization Overhead**: Identify if processes spend too much time waiting at barriers 