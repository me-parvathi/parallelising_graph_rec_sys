#!/usr/bin/env python3
import os
import json
import glob
from datetime import datetime

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def calculate_average_time(times):
    return sum(times) / len(times) if times else 0

def format_speedup(value):
    if value == 'N/A' or value is None:
        return 'N/A'
    return f"{float(value):.2f}x"

def generate_summary():
    # Find all benchmark result directories
    benchmark_dirs = glob.glob("benchmark_results_*")
    if not benchmark_dirs:
        print("No benchmark results found!")
        return

    # Sort by timestamp (newest first)
    benchmark_dirs.sort(reverse=True)
    latest_dir = benchmark_dirs[0]

    # Create summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"benchmark_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== Benchmark Summary Report ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Process each dataset
        datasets = ["Cora", "Citeseer", "PubMed"]
        for dataset in datasets:
            dataset_dir = os.path.join(latest_dir, dataset)
            if not os.path.exists(dataset_dir):
                continue
                
            f.write(f"\n=== {dataset} Dataset ===\n")
            
            # Load speedup analysis
            speedup_file = os.path.join(dataset_dir, "speedup_analysis_world2.json")
            speedup_data = load_json_file(speedup_file)
            
            if speedup_data:
                f.write("\nSpeedup Analysis:\n")
                f.write(f"Average Speedup: {format_speedup(speedup_data.get('average_speedup'))}\n")
                f.write(f"Best Speedup: {format_speedup(speedup_data.get('best_speedup'))}\n")
                f.write(f"Worst Speedup: {format_speedup(speedup_data.get('worst_speedup'))}\n")
                
                # Write epoch-wise speedups
                f.write("\nEpoch-wise Speedups:\n")
                for epoch, speedup in speedup_data.get('epoch_speedups', {}).items():
                    f.write(f"Epoch {epoch}: {format_speedup(speedup)}\n")
            
            # Load benchmark summary
            summary_file = os.path.join(dataset_dir, "benchmark_summary.json")
            summary_data = load_json_file(summary_file)
            
            if summary_data:
                f.write("\nTiming Information:\n")
                
                # Sequential times
                seq_times = [run['total_time'] for run in summary_data.get('sequential', [])]
                avg_seq_time = calculate_average_time(seq_times)
                f.write(f"Sequential (avg): {avg_seq_time:.2f} seconds\n")
                
                # Parallel times
                parallel_data = summary_data.get('parallel', {}).get('2', [])
                if parallel_data:
                    parallel_times = [run['total_time'] for run in parallel_data]
                    avg_parallel_time = calculate_average_time(parallel_times)
                    f.write(f"Parallel (2 GPUs, avg): {avg_parallel_time:.2f} seconds\n")
                    
                    # Calculate actual speedup
                    if avg_seq_time > 0:
                        actual_speedup = avg_seq_time / avg_parallel_time
                        f.write(f"Measured Speedup: {actual_speedup:.2f}x\n")
            
            f.write("\n" + "="*50 + "\n")
    
    print(f"Summary report generated: {summary_file}")

if __name__ == "__main__":
    generate_summary() 