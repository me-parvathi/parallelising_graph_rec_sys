#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
from typing import List, Dict
import json
from profiling import compare_performance, strong_scaling_analysis

def run_benchmark(
    dataset: str = "Cora",
    world_sizes: List[int] = [1, 2],
    epochs: int = 50,
    repeat: int = 3,
    output_dir: str = "benchmark_results"
) -> Dict:
    """
    Run benchmarks for both sequential and parallel implementations with multiple world sizes.
    
    Args:
        dataset: Dataset to use (Cora, Citeseer, PubMed)
        world_sizes: List of world sizes to test
        epochs: Number of epochs to run
        repeat: Number of times to repeat each benchmark
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter world sizes based on available GPUs
    available_gpus = get_gpu_count()
    filtered_world_sizes = [ws for ws in world_sizes if ws <= available_gpus or ws == 1]
    
    if len(filtered_world_sizes) < len(world_sizes):
        print(f"\n⚠️ WARNING: Some requested world sizes exceed available GPUs ({available_gpus}).")
        print(f"Proceeding with world sizes: {filtered_world_sizes}")
        world_sizes = filtered_world_sizes
    
    results = {
        "dataset": dataset,
        "epochs": epochs,
        "repeat": repeat,
        "sequential": [],
        "parallel": {}
    }
    
    # Run sequential benchmarks
    print(f"\n=== Running sequential benchmarks ({repeat} runs) ===")
    for i in range(repeat):
        print(f"Sequential run {i+1}/{repeat}")
        log_file = f"{output_dir}/sequential_{dataset}_{i+1}.csv"
        cmd = [
            "python", "main.py",
            "--dataset", dataset,
            "--epochs", str(epochs),
            "--profile", "--track_memory", "--track_time",
            "--log_file", log_file
        ]
        
        start_time = time.time()
        subprocess.run(cmd, check=True)
        total_time = time.time() - start_time
        
        results["sequential"].append({
            "run": i+1,
            "log_file": log_file,
            "total_time": total_time
        })
    
    # Run parallel benchmarks for each world size
    for world_size in world_sizes:
        if world_size == 1:
            continue  # Skip world_size=1 as it's already covered by sequential
            
        results["parallel"][world_size] = []
        print(f"\n=== Running parallel benchmarks with world_size={world_size} ({repeat} runs) ===")
        
        for i in range(repeat):
            print(f"Parallel run with world_size={world_size}, run {i+1}/{repeat}")
            log_file = f"{output_dir}/parallel_{dataset}_world{world_size}_{i+1}.csv"
            cmd = [
                "python", "main.py",
                "--dataset", dataset,
                "--epochs", str(epochs),
                "--parallel", "nb_bsp",
                "--world_size", str(world_size),
                "--profile", "--track_memory", "--track_time",
                "--log_file", log_file
            ]
            
            start_time = time.time()
            try:
                subprocess.run(cmd, check=True)
                success = True
            except subprocess.CalledProcessError:
                print(f"Error: Benchmark failed for world_size={world_size}, run {i+1}")
                print(f"This may be due to insufficient GPUs (required: {world_size}, available: {get_gpu_count()})")
                success = False
            total_time = time.time() - start_time
            
            if success:
                results["parallel"][world_size].append({
                    "run": i+1,
                    "log_file": log_file,
                    "total_time": total_time
                })
            else:
                # Remove this world size from further runs to avoid repeated failures
                print(f"Skipping remaining runs for world_size={world_size}")
                break
        
        # If all runs failed, remove this world size from results
        if not results["parallel"][world_size]:
            del results["parallel"][world_size]
    
    # Save benchmark summary
    with open(f"{output_dir}/benchmark_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def get_gpu_count():
    """Get the number of available GPUs."""
    try:
        # Try to import torch and get device count
        import torch
        return torch.cuda.device_count()
    except:
        # If torch is not available or fails, return 0
        return 0

def analyze_benchmark_results(
    results_dir: str = "benchmark_results",
    dataset: str = "Cora"
) -> None:
    """
    Analyze benchmark results and generate performance comparison reports.
    
    Args:
        results_dir: Directory containing benchmark results
        dataset: Dataset used for benchmarking
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Find the best (fastest) run for each configuration
    sequential_logs = [f for f in os.listdir(results_dir) if f.startswith(f"sequential_{dataset}")]
    
    if not sequential_logs:
        print(f"No sequential benchmark results found for dataset {dataset}")
        return
    
    # Use the first sequential run as baseline
    sequential_log = os.path.join(results_dir, sequential_logs[0])
    
    # Find parallel logs for different world sizes
    parallel_logs = {}
    for f in os.listdir(results_dir):
        if f.startswith(f"parallel_{dataset}_world"):
            try:
                # Extract world size from filename
                parts = f.split("_")
                for part in parts:
                    if part.startswith("world"):
                        world_size = int(part[5:])
                        # Check if file exists and is not empty
                        file_path = os.path.join(results_dir, f)
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            parallel_logs[world_size] = file_path
                        break
            except:
                continue
    
    if not parallel_logs:
        print(f"No parallel benchmark results found for dataset {dataset}")
        return
    
    # Generate one-to-one comparisons for each world size
    for world_size, parallel_log in parallel_logs.items():
        print(f"\n=== Analyzing performance: Sequential vs World Size {world_size} ===")
        output_file = os.path.join(results_dir, f"speedup_analysis_world{world_size}.json")
        plot_file = os.path.join(results_dir, f"speedup_plot_world{world_size}.png")
        
        try:
            compare_performance(
                parallel_log_file=parallel_log,
                sequential_log_file=sequential_log,
                output_file=output_file,
                plot_file=plot_file
            )
        except Exception as e:
            print(f"Error analyzing world_size={world_size}: {str(e)}")
    
    # Perform scaling analysis across all world sizes if we have more than one valid result
    valid_world_sizes = list(parallel_logs.keys())
    if len(valid_world_sizes) > 0:
        print("\n=== Performing scaling analysis across all world sizes ===")
        # Add sequential run as world_size=1
        scaling_logs = {1: sequential_log}
        scaling_logs.update(parallel_logs)
        
        output_file = os.path.join(results_dir, f"scaling_analysis_{dataset}.json")
        plot_file = os.path.join(results_dir, f"scaling_plot_{dataset}.png")
        
        try:
            strong_scaling_analysis(
                log_files=scaling_logs,
                output_file=output_file,
                plot_file=plot_file
            )
        except Exception as e:
            print(f"Error performing scaling analysis: {str(e)}")
    else:
        print("Not enough valid world sizes for scaling analysis")

def generate_report(
    results_dir: str = "benchmark_results",
    dataset: str = "Cora",
    output_file: str = "speedup_report.md"
) -> None:
    """
    Generate a comprehensive Markdown report of parallel speedup results.
    
    Args:
        results_dir: Directory containing benchmark results
        dataset: Dataset used for benchmarking
        output_file: Path to save the report
    """
    # Collect all analysis files
    speedup_files = [f for f in os.listdir(results_dir) if f.startswith("speedup_analysis")]
    scaling_file = next((f for f in os.listdir(results_dir) if f.startswith("scaling_analysis")), None)
    
    if not speedup_files:
        print("No speedup analysis files found")
        return
    
    # Start the report
    report = [
        f"# Parallel Speedup Analysis Report for {dataset}",
        "",
        "## Overview",
        "",
        "This report documents the performance improvements achieved by using parallel processing for graph-based recommendation systems.",
        "",
        "### System Information",
        "",
        "- Dataset: " + dataset,
        f"- Number of parallel configurations tested: {len(speedup_files)}",
        "- Date: " + time.strftime("%Y-%m-%d"),
        "",
        "## Speedup Summary",
        "",
    ]
    
    # Add speedup summary for each world size
    for speedup_file in sorted(speedup_files):
        try:
            with open(os.path.join(results_dir, speedup_file), 'r') as f:
                data = json.load(f)
                
                world_size = data.get("parallel_world_size", "unknown")
                speedups = data.get("speedups", {})
                
                report.extend([
                    f"### World Size: {world_size}",
                    "",
                    f"- Overall Speedup: {speedups.get('epoch_time', 0):.2f}x",
                    f"- Training Time Speedup: {speedups.get('train_time', 0):.2f}x",
                    f"- Evaluation Time Speedup: {speedups.get('eval_time', 0):.2f}x",
                    "",
                    f"![Speedup Chart](speedup_plot_world{world_size}.png)",
                    "",
                    "#### Performance Metrics",
                    "",
                    "| Metric | Sequential | Parallel | Ratio |",
                    "|--------|------------|----------|-------|",
                ])
                
                # Add detailed metrics
                seq_metrics = data.get("sequential_avg_metrics", {})
                par_metrics = data.get("parallel_avg_metrics", {})
                
                for metric in ["epoch_time", "train_time", "eval_time", "gpu_mem", "cpu_mem"]:
                    if metric in seq_metrics and metric in par_metrics:
                        seq_val = seq_metrics[metric]
                        par_val = par_metrics[metric]
                        ratio = seq_val / par_val if par_val > 0 else 0
                        
                        # Format based on metric type
                        if "time" in metric:
                            report.append(f"| {metric} | {seq_val:.4f}s | {par_val:.4f}s | {ratio:.2f}x |")
                        else:
                            report.append(f"| {metric} | {seq_val:.2f}MB | {par_val:.2f}MB | {ratio:.2f}x |")
                
                report.append("")
                
                # Add accuracy comparison
                acc_diff = data.get("accuracy_diff", {})
                report.extend([
                    "#### Accuracy Impact",
                    "",
                    f"- Training Accuracy Difference: {acc_diff.get('final_train_acc', 0):.4f}",
                    f"- Validation Accuracy Difference: {acc_diff.get('final_val_acc', 0):.4f}",
                    f"- Test Accuracy Difference: {acc_diff.get('final_test_acc', 0):.4f}",
                    "",
                    "> Note: Positive values indicate higher accuracy in parallel implementation.",
                    "",
                    f"![Time Comparison](epoch_time_comparison_speedup_plot_world{world_size}.png)",
                    "",
                ])
        except Exception as e:
            report.append(f"Error processing {speedup_file}: {e}")
            report.append("")
    
    # Add scaling analysis if available
    if scaling_file:
        try:
            with open(os.path.join(results_dir, scaling_file), 'r') as f:
                scaling_data = json.load(f)
                
                report.extend([
                    "## Scaling Analysis",
                    "",
                    f"Baseline: World Size {scaling_data.get('baseline_world_size', 1)}",
                    "",
                    "| World Size | Speedup | Parallel Efficiency |",
                    "|------------|---------|---------------------|",
                ])
                
                scaling = scaling_data.get("scaling", {})
                for world_size, data in sorted(scaling.items(), key=lambda x: int(x[0])):
                    speedup = data.get("speedup", 0)
                    efficiency = data.get("efficiency", 0)
                    report.append(f"| {world_size} | {speedup:.2f}x | {efficiency:.2f} |")
                
                report.extend([
                    "",
                    "![Scaling Plot](scaling_plot_" + dataset + ".png)",
                    "",
                    "![Amdahl's Law Fit](amdahls_fit_scaling_plot_" + dataset + ".png)",
                    "",
                ])
        except Exception as e:
            report.append(f"Error processing scaling analysis: {e}")
            report.append("")
    
    # Add conclusion
    report.extend([
        "## Conclusion",
        "",
        "The parallel implementation provides significant speedups over the sequential implementation, especially for larger world sizes.",
        "However, there is a trade-off between parallelization overhead and speedup gains as we increase the number of processes.",
        "",
        "For this particular dataset and model architecture, the optimal configuration appears to be a world size of [INSERT BEST WORLD SIZE],",
        "which provides the best balance between speedup and resource utilization.",
        "",
        "## Next Steps",
        "",
        "- Test with larger datasets to evaluate scalability",
        "- Optimize communication patterns for better efficiency",
        "- Investigate hybrid parallelism (data + model parallelism)",
        "",
    ])
    
    # Write the report
    with open(os.path.join(results_dir, output_file), 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {os.path.join(results_dir, output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel benchmarks and analyze results")
    parser.add_argument("--dataset", type=str, default="Cora", choices=["Cora", "Citeseer", "PubMed", "Reddit"],
                        help="Dataset to use for benchmarking")
    parser.add_argument("--world_sizes", type=int, nargs="+", default=[1, 2],
                        help="List of world sizes to benchmark")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for each benchmark run")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of times to repeat each benchmark")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--analyze_only", action="store_true",
                        help="Only analyze existing results, don't run new benchmarks")
    parser.add_argument("--report", action="store_true",
                        help="Generate a comprehensive report")
    parser.add_argument("--auto_limit", action="store_true", default=True,
                        help="Automatically limit world sizes based on available GPUs")
    
    args = parser.parse_args()
    
    # Automatically filter world sizes if requested
    if args.auto_limit:
        available_gpus = get_gpu_count()
        original_sizes = args.world_sizes.copy()
        args.world_sizes = [ws for ws in args.world_sizes if ws <= available_gpus or ws == 1]
        if len(args.world_sizes) < len(original_sizes):
            print(f"\n⚠️ WARNING: Limited world sizes to match available GPUs ({available_gpus}).")
            print(f"Using world sizes: {args.world_sizes}")
    
    if not args.analyze_only:
        # Run benchmarks
        results = run_benchmark(
            dataset=args.dataset,
            world_sizes=args.world_sizes,
            epochs=args.epochs,
            repeat=args.repeat,
            output_dir=args.output_dir
        )
    
    # Analyze results
    analyze_benchmark_results(
        results_dir=args.output_dir,
        dataset=args.dataset
    )
    
    # Generate report if requested
    if args.report:
        generate_report(
            results_dir=args.output_dir,
            dataset=args.dataset,
            output_file=f"speedup_report_{args.dataset}.md"
        ) 