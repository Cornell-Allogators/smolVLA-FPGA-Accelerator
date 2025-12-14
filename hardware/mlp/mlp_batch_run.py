"""
MLP Batch Runner - Test Multiple Configurations

Tests different Allo MLP configurations:
- With/without dataflow
- Different parallelism factors
- Extracts synthesis reports (resource usage, latency)
- Saves logs and reports to organized folders
"""

import allo
import numpy as np
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from allo.ir.types import int8

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mlp_scheduler import schedule_mlp


def extract_csynth_report(project_path):
    """
    Extract key metrics from Vitis HLS C synthesis report.
    
    Returns dict with:
    - LUT, FF, DSP, BRAM usage
    - Latency (min, max, avg)
    - Clock period, target clock
    """
    csynth_xml = Path(project_path) / "solution1" / "syn" / "report" / "csynth.xml"
    
    if not csynth_xml.exists():
        return None
    
    try:
        tree = ET.parse(csynth_xml)
        root = tree.getroot()
        
        metrics = {}
        
        # Extract resource usage
        area_estimates = root.find(".//AreaEstimates")
        if area_estimates is not None:
            resources = area_estimates.find(".//Resources")
            if resources is not None:
                metrics['LUT'] = resources.findtext('LUT', 'N/A')
                metrics['FF'] = resources.findtext('FF', 'N/A')
                metrics['DSP'] = resources.findtext('DSP', 'N/A')
                metrics['BRAM'] = resources.findtext('BRAM_18K', 'N/A')
        
        # Extract performance estimates
        perf_estimates = root.find(".//PerformanceEstimates")
        if perf_estimates is not None:
            timing = perf_estimates.find(".//SummaryOfTimingAnalysis")
            if timing is not None:
                metrics['EstimatedClockPeriod'] = timing.findtext('EstimatedClockPeriod', 'N/A')
                metrics['TargetClockPeriod'] = timing.findtext('TargetClockPeriod', 'N/A')
            
            # Extract latency
            summary = perf_estimates.find(".//SummaryOfOverallLatency")
            if summary is not None:
                metrics['LatencyMin'] = summary.findtext('Best-caseLatency', 'N/A')
                metrics['LatencyMax'] = summary.findtext('Worst-caseLatency', 'N/A')
                metrics['LatencyAvg'] = summary.findtext('Average-caseLatency', 'N/A')
        
        return metrics
    
    except Exception as e:
        print(f"  Warning: Failed to parse csynth.xml: {e}")
        return None


def save_report(config_name, project_path, output_dir, metrics):
    """Save synthesis report and logs to output directory."""
    config_dir = Path(output_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = config_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Copy csynth report if exists
    csynth_xml = Path(project_path) / "solution1" / "syn" / "report" / "csynth.xml"
    if csynth_xml.exists():
        shutil.copy(csynth_xml, config_dir / "csynth.xml")
    
    csynth_rpt = Path(project_path) / "solution1" / "syn" / "report" / "csynth.rpt"
    if csynth_rpt.exists():
        shutil.copy(csynth_rpt, config_dir / "csynth.rpt")
    
    # Copy HLS log
    hls_log = Path(project_path) / "vitis_hls.log"
    if hls_log.exists():
        shutil.copy(hls_log, config_dir / "vitis_hls.log")
    
    print(f"  ✓ Saved reports to: {config_dir}")


def print_metrics_table(results):
    """Print a formatted table of all results."""
    print("\n" + "=" * 120)
    print("SYNTHESIS RESULTS SUMMARY")
    print("=" * 120)
    
    # Header
    header = f"{'Configuration':<40} {'LUT':>10} {'FF':>10} {'DSP':>6} {'BRAM':>6} {'Latency (cycles)':>20}"
    print(header)
    print("-" * 120)
    
    # Results
    for config_name, metrics in results.items():
        if metrics is None:
            print(f"{config_name:<40} {'FAILED':>10}")
            continue
        
        lut = metrics.get('LUT', 'N/A')
        ff = metrics.get('FF', 'N/A')
        dsp = metrics.get('DSP', 'N/A')
        bram = metrics.get('BRAM', 'N/A')
        
        lat_min = metrics.get('LatencyMin', 'N/A')
        lat_max = metrics.get('LatencyMax', 'N/A')
        
        if lat_min != 'N/A' and lat_max != 'N/A':
            if lat_min == lat_max:
                latency_str = f"{lat_min}"
            else:
                latency_str = f"{lat_min} - {lat_max}"
        else:
            latency_str = "N/A"
        
        print(f"{config_name:<40} {lut:>10} {ff:>10} {dsp:>6} {bram:>6} {latency_str:>20}")
    
    print("=" * 120)


def create_mlp_dataflow(dtype, D, L, enable_dataflow=True):
    """Create MLP with configurable dataflow."""
    from allo.ir.types import int8, int32
    
    def mlp_dataflow(
        X: "dtype[L, D]",
        W_1: "dtype[D, 4*D]",
        B_1: "dtype[4*D]",
        W_2: "dtype[4*D, D]",
        B_2: "dtype[D]",
        gamma: "dtype[D]",
        beta: "dtype[D]",
        Y: "dtype[L, D]"
    ):
        # FC1: [L, D] @ [D, 4*D] -> [L, 4*D]
        FC1: int32[L, 4*D]
        for i, j in allo.grid(L, 4*D):
            FC1[i, j] = B_1[j]
            for k in allo.reduction(D):
                FC1[i, j] += X[i, k] * W_1[k, j]
        
        # GELU activation
        FC1_act: int32[L, 4*D]
        for i, j in allo.grid(L, 4*D):
            x: int32 = FC1[i, j]
            x_float: allo.float32 = x
            x3 = x_float * x_float * x_float
            inner = 0.7978845608028654 * (x_float + 0.044715 * x3)
            tanh_val = allo.tanh(inner)
            gelu_out = 0.5 * x_float * (1.0 + tanh_val)
            FC1_act[i, j] = gelu_out
        
        # FC2: [L, 4*D] @ [4*D, D] -> [L, D]
        FC2: int32[L, D]
        for i, j in allo.grid(L, D):
            FC2[i, j] = B_2[j]
            for k in allo.reduction(4*D):
                FC2[i, j] += FC1_act[i, k] * W_2[k, j]
        
        # LayerNorm: int32 -> int8
        total: int32[L] = 0
        total_sq: int32[L] = 0
        for i in allo.grid(L):
            for j in allo.reduction(D):
                val: int32 = FC2[i, j]
                total[i] += val
                total_sq[i] += val * val
        
        mean: allo.float32[L]
        inv_std: allo.float32[L]
        for i in allo.grid(L):
            mean_i: allo.float32 = total[i] / D
            mean[i] = mean_i
            variance: allo.float32 = (total_sq[i] / D) - (mean_i * mean_i)
            inv_std[i] = 1.0 / allo.sqrt(variance + 1e-8)
        
        for i in allo.grid(L):
            mean_i: allo.float32 = mean[i]
            inv_std_i: allo.float32 = inv_std[i]
            for j in allo.grid(D):
                norm_val: allo.float32 = (FC2[i, j] - mean_i) * inv_std_i
                scaled: allo.float32 = norm_val * gamma[j]
                shifted: allo.float32 = scaled + beta[j]
                Y[i, j] = shifted
    
    s = allo.customize(mlp_dataflow, instantiate=[dtype, D, L])
    
    if enable_dataflow:
        s.dataflow("mlp_dataflow")
    
    return s


def create_mlp_with_parallelism(dtype, D, L, parallel_factor):
    """Create MLP with specific parallelism factor."""
    from allo.ir.types import int8, int32
    
    def mlp_parallel(
        X: "dtype[L, D]",
        W_1: "dtype[D, 4*D]",
        B_1: "dtype[4*D]",
        W_2: "dtype[4*D, D]",
        B_2: "dtype[D]",
        gamma: "dtype[D]",
        beta: "dtype[D]",
        Y: "dtype[L, D]"
    ):
        # FC1: [L, D] @ [D, 4*D] -> [L, 4*D]
        FC1: int32[L, 4*D]
        for i, j in allo.grid(L, 4*D):
            FC1[i, j] = B_1[j]
            for k in allo.reduction(D):
                FC1[i, j] += X[i, k] * W_1[k, j]
        
        # GELU activation
        FC1_act: int32[L, 4*D]
        for i, j in allo.grid(L, 4*D):
            x: int32 = FC1[i, j]
            x_float: allo.float32 = x
            x3 = x_float * x_float * x_float
            inner = 0.7978845608028654 * (x_float + 0.044715 * x3)
            tanh_val = allo.tanh(inner)
            gelu_out = 0.5 * x_float * (1.0 + tanh_val)
            FC1_act[i, j] = gelu_out
        
        # FC2: [L, 4*D] @ [4*D, D] -> [L, D]
        FC2: int32[L, D]
        for i, j in allo.grid(L, D):
            FC2[i, j] = B_2[j]
            for k in allo.reduction(4*D):
                FC2[i, j] += FC1_act[i, k] * W_2[k, j]
        
        # LayerNorm: int32 -> int8
        total: int32[L] = 0
        total_sq: int32[L] = 0
        for i in allo.grid(L):
            for j in allo.reduction(D):
                val: int32 = FC2[i, j]
                total[i] += val
                total_sq[i] += val * val
        
        mean: allo.float32[L]
        inv_std: allo.float32[L]
        for i in allo.grid(L):
            mean_i: allo.float32 = total[i] / D
            mean[i] = mean_i
            variance: allo.float32 = (total_sq[i] / D) - (mean_i * mean_i)
            inv_std[i] = 1.0 / allo.sqrt(variance + 1e-8)
        
        for i in allo.grid(L):
            mean_i: allo.float32 = mean[i]
            inv_std_i: allo.float32 = inv_std[i]
            for j in allo.grid(D):
                norm_val: allo.float32 = (FC2[i, j] - mean_i) * inv_std_i
                scaled: allo.float32 = norm_val * gamma[j]
                shifted: allo.float32 = scaled + beta[j]
                Y[i, j] = shifted
    
    s = allo.customize(mlp_parallel, instantiate=[dtype, D, L])
    
    # Apply dataflow
    s.dataflow("mlp_parallel")
    
    # Apply parallelism to matmul loops
    s.pipeline("mlp_parallel", axis=1)
    s.unroll("mlp_parallel", axis=1, factor=parallel_factor)
    
    return s


def run_synthesis_test(config_tuple):
    """
    Run HLS synthesis test for a specific configuration.
    
    Args:
        config_tuple: Tuple of (config_name, schedule_spec, dtype, L, D, output_dir)
    
    Returns:
        Tuple of (config_name, metrics)
    """
    config_name, schedule_spec, dtype, L, D, output_dir = config_tuple
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {config_name}")
    print(f"{'=' * 80}")
    print(f"  Config: L={L}, D={D}")
    
    project_name = f"mlp_batch_{config_name.replace(' ', '_').replace('/', '_')}"
    
    try:
        # Call schedule_mlp directly with appropriate arguments
        project_folder = f"{project_name}.prj"
        
        # Parse schedule_spec to determine parallelism and dataflow
        kind = schedule_spec[0]
        if kind == "dataflow":
            enable_df = schedule_spec[1]
            P = 1  # Default baseline parallelism
        elif kind == "parallel":
            P = schedule_spec[1]
            enable_df = True
        else:
            raise RuntimeError(f"Unknown schedule spec: {schedule_spec}")
        
        print(f"  Running schedule_mlp with P={P}, dataflow={enable_df}")
        schedule_mlp(
            N_T=np.int8,
            A_T=int8,
            P=P,
            mode="csyn",
            enable_dataflow=enable_df,
            project=project_folder
        )
        
        print(f"  ✓ Synthesis completed: {config_name}")
        
        # Extract metrics from the nested out.prj folder
        project_path = f"{project_folder}/out.prj"
        metrics = extract_csynth_report(project_path)
        
        if metrics:
            print(f"\n  Results for {config_name}:")
            print(f"    LUT:  {metrics.get('LUT', 'N/A')}")
            print(f"    FF:   {metrics.get('FF', 'N/A')}")
            print(f"    DSP:  {metrics.get('DSP', 'N/A')}")
            print(f"    BRAM: {metrics.get('BRAM', 'N/A')}")
            
            lat_min = metrics.get('LatencyMin', 'N/A')
            lat_max = metrics.get('LatencyMax', 'N/A')
            print(f"    Latency: {lat_min} - {lat_max} cycles")
            
            # Save reports
            save_report(config_name, project_path, output_dir, metrics)
        else:
            print(f"  ⚠ Warning: Could not extract synthesis metrics for {config_name}")
        
        return (config_name, metrics)
    
    except Exception as e:
        print(f"  ✗ FAILED: {config_name} - {e}")
        import traceback
        traceback.print_exc()
        return (config_name, None)


def main():
    """Run batch synthesis tests."""
    from allo.ir.types import int8
    
    # ============ CONFIGURATION ============
    L, D = 1024, 3072  # Full-scale dimensions from vision model MLP
    DTYPE = int8
    
    # Parallelism factor range (powers of 2)
    MIN_PARALLEL = 1    # Minimum parallelism factor (power of 2)
    MAX_PARALLEL = 16384    # Maximum parallelism factor (power of 2)
    # =======================================
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"batch_results_{timestamp}"
    
    print("\n" + "█" * 80)
    print("  MLP BATCH SYNTHESIS RUNNER")
    print("█" * 80)
    print(f"\nDimensions: L={L}, D={D}")
    print(f"Data Type: {DTYPE}")
    print(f"Output Directory: {output_dir}")
    print(f"\nThis will run HLS synthesis (csyn mode) for multiple configurations.")
    print(f"Each synthesis may take 2-5 minutes.")
    
    # Define test configurations
    configs = []
    
    # 1. Baseline: No dataflow (serializable spec)
    configs.append(("baseline_no_dataflow", ("dataflow", False)))

    # 2. With dataflow
    configs.append(("with_dataflow", ("dataflow", True)))
    
    # 3. Dataflow + parallelism factors (powers of 2 from MIN to MAX)
    import math
    min_exp = int(math.log2(MIN_PARALLEL))
    max_exp = int(math.log2(MAX_PARALLEL))
    parallel_factors = [2**i for i in range(min_exp, max_exp + 1)]
    
    print(f"\nParallelism factors to test: {parallel_factors}")
    
    for pf in parallel_factors:
        configs.append((f"dataflow_parallel_{pf}x", ("parallel", pf)))
    print(f"Total configurations to test: {len(configs)}")
    
    # Determine number of parallel workers (reduced to avoid OOM)
    max_workers = min(multiprocessing.cpu_count(), len(configs))  # Limit to 2 parallel jobs to avoid OOM
    print(f"\nUsing {max_workers} parallel workers")
    print("\nStarting synthesis tests...")
    
    # Prepare config tuples for parallel execution (include dtype)
    config_tuples = [(name, spec, DTYPE, L, D, output_dir) for name, spec in configs]
    
    # Run all tests in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {executor.submit(run_synthesis_test, config_tuple): config_tuple[0] 
                           for config_tuple in config_tuples}
        
        # Process completed jobs
        completed = 0
        for future in as_completed(future_to_config):
            config_name = future_to_config[future]
            try:
                name, metrics = future.result()
                results[name] = metrics
                completed += 1
                print(f"\n[Progress: {completed}/{len(configs)}] Completed: {name}")
            except Exception as e:
                print(f"\n✗ Exception in {config_name}: {e}")
                results[config_name] = None
                completed += 1
    
    # Print summary table
    print_metrics_table(results)
    
    # Save summary JSON
    summary_file = Path(output_dir) / "summary.json"
    summary_data = {
        'timestamp': timestamp,
        'dimensions': {'L': L, 'D': D},
        'dtype': str(DTYPE),
        'results': {k: v for k, v in results.items()}
    }
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"  - Individual reports in subdirectories")
    print(f"  - Summary: {summary_file}")


if __name__ == "__main__":
    main()
