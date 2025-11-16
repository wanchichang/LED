"""
Example script showing how to use the outputs from main_led_nba.py

This script demonstrates how to:
1. Parse performance metrics from log files
2. Export metrics to CSV
3. Compare different experiments
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_log_file(log_path: str) -> Dict[str, float]:
    """
    Parse performance metrics from a log file.
    
    Args:
        log_path: Path to the log.txt file
        
    Returns:
        Dictionary with metric names as keys and values as floats
        Example: {'ADE_1s': 0.1764, 'FDE_1s': 0.2691, ...}
    """
    metrics = {}
    
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return metrics
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match lines like: --ADE(1s): 0.1764	--FDE(1s): 0.2691
            if 'ADE' in line and 'FDE' in line:
                matches = re.findall(r'(ADE|FDE)\((\d+)s\):\s+([\d.]+)', line)
                for metric_type, time_horizon, value in matches:
                    key = f"{metric_type}_{time_horizon}s"
                    metrics[key] = float(value)
    
    return metrics


def export_metrics_to_csv(log_path: str, output_csv: str):
    """
    Export metrics from log file to CSV format.
    
    Args:
        log_path: Path to input log.txt file
        output_csv: Path to output CSV file
    """
    metrics = parse_log_file(log_path)
    
    if not metrics:
        print("No metrics found in log file")
        return
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in sorted(metrics.items()):
            writer.writerow([key, value])
    
    print(f"Metrics exported to {output_csv}")


def compare_experiments(results_dir: str, output_file: Optional[str] = None):
    """
    Compare metrics from multiple experiments.
    
    Args:
        results_dir: Directory containing experiment results (e.g., 'results/led_augment')
        output_file: Optional path to save comparison as JSON
    """
    results_dir = Path(results_dir)
    experiments = {}
    
    # Find all log files
    log_files = list(results_dir.glob('*/log/log.txt'))
    
    for log_file in log_files:
        exp_name = log_file.parent.parent.name  # Get experiment name from path
        metrics = parse_log_file(str(log_file))
        if metrics:
            experiments[exp_name] = metrics
    
    # Print comparison
    print("\n" + "="*60)
    print("Experiment Comparison")
    print("="*60)
    
    if not experiments:
        print("No experiments found")
        return
    
    # Get all metric keys
    all_metrics = set()
    for exp_metrics in experiments.values():
        all_metrics.update(exp_metrics.keys())
    
    # Print header
    print(f"{'Experiment':<20}", end="")
    for metric in sorted(all_metrics):
        print(f"{metric:>12}", end="")
    print()
    print("-" * (20 + 12 * len(all_metrics)))
    
    # Print each experiment
    for exp_name, metrics in sorted(experiments.items()):
        print(f"{exp_name:<20}", end="")
        for metric in sorted(all_metrics):
            value = metrics.get(metric, "N/A")
            if isinstance(value, float):
                print(f"{value:>12.4f}", end="")
            else:
                print(f"{str(value):>12}", end="")
        print()
    
    # Save to JSON if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(experiments, f, indent=2)
        print(f"\nComparison saved to {output_file}")


def get_best_metric(results_dir: str, metric_name: str = 'ADE_4s') -> Dict:
    """
    Find the experiment with the best (lowest) metric value.
    
    Args:
        results_dir: Directory containing experiment results
        metric_name: Name of metric to compare (default: 'ADE_4s')
        
    Returns:
        Dictionary with experiment name and metric value
    """
    results_dir = Path(results_dir)
    best_exp = None
    best_value = float('inf')
    
    log_files = list(results_dir.glob('*/log/log.txt'))
    
    for log_file in log_files:
        exp_name = log_file.parent.parent.name
        metrics = parse_log_file(str(log_file))
        
        if metric_name in metrics:
            value = metrics[metric_name]
            if value < best_value:
                best_value = value
                best_exp = exp_name
    
    if best_exp:
        return {'experiment': best_exp, metric_name: best_value}
    return {}


def print_summary(log_path: str):
    """
    Print a formatted summary of metrics from a log file.
    
    Args:
        log_path: Path to log.txt file
    """
    metrics = parse_log_file(log_path)
    
    if not metrics:
        print("No metrics found")
        return
    
    print("\n" + "="*50)
    print("Performance Summary")
    print("="*50)
    
    # Group by time horizon
    time_horizons = sorted(set(re.search(r'(\d+)s', k).group(1) for k in metrics.keys() if re.search(r'(\d+)s', k)))
    
    for time in time_horizons:
        ade_key = f'ADE_{time}s'
        fde_key = f'FDE_{time}s'
        ade = metrics.get(ade_key, 'N/A')
        fde = metrics.get(fde_key, 'N/A')
        print(f"At {time}s prediction horizon:")
        print(f"  ADE: {ade:.4f}" if isinstance(ade, float) else f"  ADE: {ade}")
        print(f"  FDE: {fde:.4f}" if isinstance(fde, float) else f"  FDE: {fde}")
        print()


if __name__ == "__main__":
    # Example usage
    
    # 1. Parse a specific log file
    log_path = "results/led_augment/reproduce/log/log.txt"
    if os.path.exists(log_path):
        print_summary(log_path)
        
        # 2. Export to CSV
        export_metrics_to_csv(log_path, "results/led_augment/reproduce/metrics.csv")
        
        # 3. Parse programmatically
        metrics = parse_log_file(log_path)
        print("Extracted metrics:")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")
    
    # 4. Compare multiple experiments
    if os.path.exists("results/led_augment"):
        compare_experiments("results/led_augment", "results/led_augment/comparison.json")
        
        # 5. Find best experiment
        best = get_best_metric("results/led_augment", "ADE_4s")
        if best:
            print(f"\nBest experiment (lowest ADE_4s): {best['experiment']} with ADE_4s={best['ADE_4s']:.4f}")

