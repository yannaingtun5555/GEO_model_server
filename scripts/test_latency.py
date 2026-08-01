#!/usr/bin/env python3
"""
scripts/test_latency.py — Performance Benchmark for Serving Optimizations
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_benchmark():
    model_path = PROJECT_ROOT / "models" / "crop_yield_t_ha_rf_regressor.pkl"
    if not model_path.exists():
        # Fallback to prototype
        model_path = PROJECT_ROOT / "models_prototypes" / "crop_yield_t_ha_rf_regressor.pkl"
    
    if not model_path.exists():
        print(f"[ERROR] Could not find any model file at {model_path}!")
        sys.exit(1)

    print("=====================================================================")
    # Load Model
    print(f"[1/4] Loading model: {model_path.name}...")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = artifact["features"]
    
    # Create sample inputs
    x_vals = [0.5] * len(features)
    X_in = pd.DataFrame([x_vals], columns=features)
    
    # Store original estimators
    original_estimators = list(model.estimators_)
    
    print(f"  ✓ Loaded Random Forest Model with {len(original_estimators)} trees.")
    print("=====================================================================")

    # Test 1: Original Model (Simulated 500 Trees, n_jobs=-1)
    print("[2/4] Benchmarking Original Configuration (500 Trees, n_jobs=-1)...")
    model.estimators_ = original_estimators * (5 if len(original_estimators) <= 100 else 1)
    model.n_estimators = 500
    model.n_jobs = -1
    
    # Warmup
    model.predict(X_in)
    
    start_t = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        model.predict(X_in)[0]
    time_original_ms = (time.perf_counter() - start_t) * 1000 / iterations
    print(f"  ✓ Average Latency (Original): {time_original_ms:.3f} ms")
    
    # Test 2: Optimized Model (Pruned 100 Trees, n_jobs=1)
    print("\n[3/4] Benchmarking Optimized Configuration (100 Trees, n_jobs=1)...")
    model.estimators_ = original_estimators[:100]
    model.n_estimators = 100
    model.n_jobs = 1
    
    # Warmup
    model.predict(X_in)
    
    start_t = time.perf_counter()
    for _ in range(iterations):
        model.predict(X_in)[0]
    time_optimized_ms = (time.perf_counter() - start_t) * 1000 / iterations
    print(f"  ✓ Average Latency (Optimized): {time_optimized_ms:.3f} ms")
    
    # Test 3: Multi-Model Concurrency Simulation (19 Target predictions)
    print("\n[4/4] Benchmarking 19-Model Concurrency Simulation...")
    mock_tasks = []
    for i in range(19):
        mock_tasks.append({
            "model": model,
            "X": X_in
        })
        
    # Sequential
    start_t = time.perf_counter()
    for task in mock_tasks:
        task["model"].predict(task["X"])[0]
    time_seq_ms = (time.perf_counter() - start_t) * 1000
    print(f"  ✓ 19 Sequential Predictions: {time_seq_ms:.3f} ms")
    
    # Parallel (ThreadPoolExecutor)
    executor = ThreadPoolExecutor(max_workers=4)
    def _run_pred(task):
        return task["model"].predict(task["X"])[0]
        
    start_t = time.perf_counter()
    list(executor.map(_run_pred, mock_tasks))
    time_para_ms = (time.perf_counter() - start_t) * 1000
    print(f"  ✓ 19 Parallel (4 Threads) Predictions: {time_para_ms:.3f} ms")

    print("=====================================================================")
    print("                      BENCHMARK SUMMARY RESULTS                      ")
    print("=====================================================================")
    speedup_factor = time_original_ms / time_optimized_ms
    print(f"Single-Model Speedup   : {speedup_factor:.2f}x faster")
    print(f"Single-Model Latency   : {time_original_ms:.2f} ms → {time_optimized_ms:.2f} ms")
    print(f"Memory Reduction Est.  : ~80% RAM footprint savings (500 → 100 trees)")
    print(f"Multi-Model Latency    : ~{time_seq_ms/1000:.2f}s sequential → ~{time_para_ms/1000:.2f}s parallel")
    print("=====================================================================")

if __name__ == "__main__":
    run_benchmark()
