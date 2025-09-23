#!/usr/bin/env python3
"""
Project: OCTAX - Accelerated CHIP-8 Arcade RL Environments for JAX.
Author: Waris Radji, Thomas Michel, Hector Piteau
Date: 2025-09-23
Copyright: MIT License - see LICENSE file for details.
Details: Please cite the project if you use this module in your research.

Description:
    This module provides a standalone GPU memory profiler for Python algorithms.
    It can be used as a context manager or imported as a module.

Usage:
    # As a context manager
    with GPUMemoryProfiler() as profiler:
        your_algorithm()
    profiler.print_summary()
    
    # As a decorator
    @profile_gpu_memory
    def my_function():
        your_algorithm()
    
    # Manual control
    profiler = GPUMemoryProfiler()
    profiler.start()
    your_algorithm()
    stats = profiler.stop()
"""

import subprocess
import threading
import time
import json
from typing import Optional, Dict, Any
from functools import wraps
import numpy as np

def get_gpu_memory_usage(gpu_id: int = 0) -> Optional[int]:
    """Get current GPU memory usage in MB."""
    try:
        cmd = [
            'nvidia-smi', 
            f'--id={gpu_id}',
            '--query-gpu=memory.used', 
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError):
        pass
    return None


def get_gpu_memory_total(gpu_id: int = 0) -> Optional[int]:
    """Get total GPU memory in MB."""
    try:
        cmd = [
            'nvidia-smi', 
            f'--id={gpu_id}',
            '--query-gpu=memory.total', 
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError):
        pass
    return None

class GPUMemoryProfiler:
    """Profiler for GPU memory usage."""
    
    def __init__(self, interval: float = 0.01, gpu_id: int = 0):
        """
        Initialize the profiler.
        
        Args:
            interval(float): Sampling interval in seconds (default: 0.01s = 10ms)
            gpu_id(int): GPU device ID to monitor (default: 0)
        """
        self.interval = interval
        self.gpu_id = gpu_id
        self.monitoring = False
        self.memory_usage = []
        self.timestamps = []
        self.thread = None
        self.start_time = None

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        try:
            cmd = [
                'nvidia-smi', 
                f'--id={self.gpu_id}',
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', 
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'name': values[0],
                    'memory_total_mb': int(values[1]),
                    'memory_used_mb': int(values[2]),
                    'memory_free_mb': int(values[3]),
                    'gpu_utilization_percent': int(values[4]),
                    'temperature_c': int(values[5])
                }
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, subprocess.CalledProcessError, IndexError):
            pass
        return {}
    
    def _monitor(self):
        """Internal monitoring loop."""
        while self.monitoring:
            usage = get_gpu_memory_usage(self.gpu_id)
            if usage is not None:
                current_time = time.perf_counter()
                self.memory_usage.append(usage)
                self.timestamps.append(current_time - self.start_time)
            time.sleep(self.interval)
    
    def start(self) -> 'GPUMemoryProfiler':
        """Start monitoring."""
        self.memory_usage = []
        self.timestamps = []
        self.monitoring = True
        self.start_time = time.perf_counter()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self
    
    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)  # Don't wait forever
        
        if not self.memory_usage:
            return None
        
        memory_array = np.array(self.memory_usage)
        
        return {
            'peak_memory_mb': int(np.max(memory_array)),
            'min_memory_mb': int(np.min(memory_array)),
            'avg_memory_mb': float(np.mean(memory_array)),
            'std_memory_mb': float(np.std(memory_array)),
            'median_memory_mb': float(np.median(memory_array)),
            'memory_samples': len(self.memory_usage),
            'duration_seconds': max(self.timestamps) if self.timestamps else 0,
            'sampling_rate_hz': len(self.memory_usage) / max(self.timestamps) if self.timestamps and max(self.timestamps) > 0 else 0,
            'memory_trace': self.memory_usage.copy(),
            'timestamps': self.timestamps.copy()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def print_summary(self, stats: Optional[Dict[str, Any]] = None):
        """Print a formatted summary of memory usage."""
        if stats is None:
            if hasattr(self, '_last_stats'):
                stats = self._last_stats
            else:
                print("No memory statistics available. Did you call stop()?")
                return
        
        self._last_stats = stats  # Store for later use
        
        print(f"\n=== GPU MEMORY PROFILE SUMMARY ===")
        print(f"GPU ID: {self.gpu_id}")
        print(f"Duration: {stats['duration_seconds']:.3f}s")
        print(f"Samples: {stats['memory_samples']} (avg {stats['sampling_rate_hz']:.1f} Hz)")
        print(f"\n--- Memory Usage (MB) ---")
        print(f"Peak:    {stats['peak_memory_mb']:,}")
        print(f"Average: {stats['avg_memory_mb']:,.1f}")
        print(f"Median:  {stats['median_memory_mb']:,.1f}")
        print(f"Min:     {stats['min_memory_mb']:,}")
        print(f"Std Dev: {stats['std_memory_mb']:,.1f}")
        
        # Get current GPU info for context
        gpu_info = self.get_gpu_info()
        if gpu_info:
            total_memory = gpu_info['memory_total_mb']
            peak_percent = (stats['peak_memory_mb'] / total_memory) * 100
            print(f"\n--- GPU Context ---")
            print(f"GPU: {gpu_info['name']}")
            print(f"Total Memory: {total_memory:,} MB")
            print(f"Peak Utilization: {peak_percent:.1f}%")
    
    def save_trace(self, filename: str, stats: Optional[Dict[str, Any]] = None):
        """Save detailed memory trace to file."""
        if stats is None:
            if hasattr(self, '_last_stats'):
                stats = self._last_stats
            else:
                print("No memory statistics available to save.")
                return
        
        # Prepare data for JSON serialization
        trace_data = {
            'gpu_id': self.gpu_id,
            'profiler_config': {
                'interval': self.interval,
                'sampling_rate_hz': stats['sampling_rate_hz']
            },
            'summary': {k: v for k, v in stats.items() if k not in ['memory_trace', 'timestamps']},
            'trace': [
                {'time': t, 'memory_mb': m} 
                for t, m in zip(stats['timestamps'], stats['memory_trace'])
            ],
            'gpu_info': self.get_gpu_info()
        }
        
        with open(filename, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"Memory trace saved to: {filename}")


def profile_gpu_memory(interval: float = 0.01, gpu_id: int = 0, print_summary: bool = True):
    """Decorator for profiling GPU memory usage of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = GPUMemoryProfiler(interval=interval, gpu_id=gpu_id)
            with profiler:
                result = func(*args, **kwargs)
            
            stats = profiler.stop()
            if print_summary and stats:
                print(f"\n=== Memory Profile for {func.__name__} ===")
                profiler.print_summary(stats)
            
            return result
        return wrapper
    return decorator




class GPUMemoryMonitor:
    """Monitor GPU memory usage during execution."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.monitoring = False
        self.memory_usage = []
        self.thread = None

    def _monitor(self):
        """Internal monitoring loop."""
        while self.monitoring:
            usage = get_gpu_memory_usage()
            if usage is not None:
                self.memory_usage.append(usage)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.memory_usage = []
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.thread:
            self.thread.join()

        if not self.memory_usage:
            return None

        return {
            "peak_memory_mb": max(self.memory_usage),
            "min_memory_mb": min(self.memory_usage),
            "avg_memory_mb": np.mean(self.memory_usage),
            "memory_samples": len(self.memory_usage),
        }