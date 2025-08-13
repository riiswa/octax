"""Console logging utilities for Octax training and evaluation.

This module provides a flexible logging system with callbacks and formatters
for better visibility into training loops and emulator state. Includes real-time
progress bars for JAX computations using io_callback.
"""

import time
import sys
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from functools import wraps
import jax.numpy as jnp
import jax
from jax.experimental import io_callback

from tqdm import tqdm


class ConsoleLogger:
    """Flexible console logger with callback system and formatters."""

    def __init__(
        self,
        name: str = "Octax",
        log_level: str = "INFO",
        use_colors: bool = True,
        show_timestamps: bool = True,
    ):
        self.name = name
        self.log_level = log_level.upper()
        self.use_colors = (
            use_colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )
        self.show_timestamps = show_timestamps
        self.start_time = time.time()

        self.colors = (
            {
                "DEBUG": "\033[36m",
                "INFO": "\033[32m",
                "WARNING": "\033[33m",
                "ERROR": "\033[31m",
                "CRITICAL": "\033[35m",
                "RESET": "\033[0m"
            }
            if self.use_colors
            else {
                k: ""
                for k in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "RESET"]
            }
        )

        self.level_order = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on current log level."""
        return self.level_order.get(level.upper(), 1) >= self.level_order.get(
            self.log_level, 1
        )

    def _format_message(self, level: str, message: str) -> str:
        """Format log message with timestamp, level, and colors."""
        timestamp = (
            f"[{time.time() - self.start_time:8.2f}s]" if self.show_timestamps else ""
        )
        level_str = f"[{level:>8s}]"
        name_str = f"[{self.name}]"

        if self.use_colors:
            color = self.colors.get(level.upper(), "")
            reset = self.colors["RESET"]
            level_str = f"{color}{level_str}{reset}"

        return f"{timestamp}{level_str}{name_str} {message}"

    def log(self, level: str, message: str):
        """Log a message at the specified level."""
        if self._should_log(level):
            formatted = self._format_message(level, message)
            print(formatted, flush=True)

    def debug(self, message: str):
        """Log debug message."""
        self.log("DEBUG", message)

    def info(self, message: str):
        """Log info message."""
        self.log("INFO", message)

    def warning(self, message: str):
        """Log warning message."""
        self.log("WARNING", message)

    def error(self, message: str):
        """Log error message."""
        self.log("ERROR", message)

    def critical(self, message: str):
        """Log critical message."""
        self.log("CRITICAL", message)


class TrainingLogger(ConsoleLogger):
    """Specialized logger for training loops with metrics tracking."""

    def __init__(self, name: str = "Training", **kwargs):
        super().__init__(name, **kwargs)
        self.metrics_history = []
        self.last_log_time = time.time()

    def log_training_start(self, config: Dict[str, Any]):
        """Log training configuration and start message."""
        self.info("=" * 60)
        self.info(f"Starting training with configuration:")
        for key, value in config.items():
            if isinstance(value, float):
                self.info(
                    f"  {key}: {value:.2e}" if value < 0.01 else f"  {key}: {value:.4f}"
                )
            else:
                self.info(f"  {key}: {value}")
        self.info("=" * 60)

    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        total_steps: int,
        log_interval: int = 10,
    ):
        """Log training step with metrics."""
        current_time = time.time()

        if step % log_interval == 0 or step == total_steps - 1:
            progress = (step + 1) / total_steps
            elapsed = current_time - self.start_time
            eta = elapsed / progress - elapsed if progress > 0 else 0

            metric_strs = []
            for key, value in metrics.items():
                if isinstance(value, (jnp.ndarray, float)):
                    if isinstance(value, jnp.ndarray) and value.ndim > 0:
                        val = float(jnp.mean(value))
                    else:
                        val = float(value)
                    if abs(val) < 0.01 and val != 0:
                        metric_strs.append(f"{key}={val:.2e}")
                    else:
                        metric_strs.append(f"{key}={val:.4f}")
                else:
                    metric_strs.append(f"{key}={value}")

            bar_length = 20
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            step_msg = (
                f"Step {step+1:4d}/{total_steps} "
                f"[{bar}] {progress*100:5.1f}% "
                f"ETA: {eta/60:4.1f}m | " + " | ".join(metric_strs)
            )

            self.info(step_msg)

            self.metrics_history.append(
                {"step": step, "time": current_time, "metrics": metrics.copy()}
            )

    def log_training_end(self, final_metrics: Dict[str, Any]):
        """Log training completion with final metrics."""
        elapsed = time.time() - self.start_time
        self.info("=" * 60)
        self.info(f"Training completed in {elapsed/60:.2f} minutes ({elapsed:.1f}s)")
        self.info("Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (jnp.ndarray, float)):
                # Handle multidimensional arrays by taking mean
                if isinstance(value, jnp.ndarray) and value.ndim > 0:
                    val = float(jnp.mean(value))
                else:
                    val = float(value)
                if abs(val) < 0.01 and val != 0:
                    self.info(f"  {key}: {val:.2e}")
                else:
                    self.info(f"  {key}: {val:.4f}")
            else:
                self.info(f"  {key}: {value}")
        self.info("=" * 60)


class LoggingCallback:
    """Base class for logging callbacks."""

    def on_training_start(self, config: Dict[str, Any]):
        """Called at training start."""
        pass

    def on_step(self, step: int, metrics: Dict[str, Any], state: Any = None):
        """Called at each training step."""
        pass

    def on_training_end(self, final_metrics: Dict[str, Any]):
        """Called at training end."""
        pass


class ConsoleCallback(LoggingCallback):
    """Console logging callback."""

    def __init__(self, log_interval: int = 10, logger: Optional[TrainingLogger] = None):
        self.log_interval = log_interval
        self.logger = logger or TrainingLogger()
        self.total_steps = None

    def on_training_start(self, config: Dict[str, Any]):
        self.total_steps = config.get("NUM_UPDATES", config.get("total_steps", 1000))
        self.logger.log_training_start(config)

    def on_step(self, step: int, metrics: Dict[str, Any], state: Any = None):
        if self.total_steps:
            self.logger.log_training_step(
                step, metrics, self.total_steps, self.log_interval
            )

    def on_training_end(self, final_metrics: Dict[str, Any]):
        self.logger.log_training_end(final_metrics)


class MetricsCallback(LoggingCallback):
    """Callback for tracking and computing metrics statistics."""

    def __init__(self, track_keys: List[str] = None):
        self.track_keys = track_keys or ["score", "reward", "loss"]
        self.metrics_buffer = {key: [] for key in self.track_keys}
        self.step_count = 0

    def on_step(self, step: int, metrics: Dict[str, Any], state: Any = None):
        self.step_count += 1

        for key in self.track_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, jnp.ndarray):
                    value = float(jnp.mean(value))
                elif isinstance(value, (int, float)):
                    value = float(value)
                else:
                    continue
                self.metrics_buffer[key].append(value)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get running statistics for tracked metrics."""
        stats = {}
        for key, values in self.metrics_buffer.items():
            if values:
                stats[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1],
                    "count": len(values),
                }
        return stats


def with_logging(callbacks: List[LoggingCallback] = None, config_key: str = "config"):
    """Decorator to add logging to training functions."""
    if callbacks is None:
        callbacks = [ConsoleCallback()]

    def decorator(train_fn):
        @wraps(train_fn)
        def wrapper(*args, **kwargs):
            if config_key in kwargs:
                config = kwargs[config_key]
            elif args and isinstance(args[0], dict):
                config = args[0]
            else:
                config = {}

            for callback in callbacks:
                callback.on_training_start(config)

            original_result = train_fn(*args, **kwargs)

            if isinstance(original_result, dict) and "metrics" in original_result:
                final_metrics = jax.tree.map(
                    lambda x: (
                        x[-1] if hasattr(x, "__len__") and len(x.shape) > 0 else x
                    ),
                    original_result["metrics"],
                )
                final_metrics = {
                    k: float(jnp.mean(v)) if isinstance(v, jnp.ndarray) else v
                    for k, v in final_metrics.items()
                }

                for callback in callbacks:
                    callback.on_training_end(final_metrics)

            return original_result

        return wrapper

    return decorator


def create_training_wrapper(callbacks: List[LoggingCallback] = None):
    """Create a training loop wrapper with logging callbacks."""
    if callbacks is None:
        callbacks = [ConsoleCallback(), MetricsCallback()]

    def training_wrapper(train_fn, config: Dict[str, Any]):
        """Wrap a training function with logging callbacks."""

        for callback in callbacks:
            callback.on_training_start(config)

        def logged_train_fn(rng):
            # Get original scan function and wrap it
            result = train_fn(rng)

            if isinstance(result, dict) and "metrics" in result:
                metrics_data = result["metrics"]

                if hasattr(metrics_data, "__len__"):
                    for step in range(len(next(iter(metrics_data.values())))):
                        step_metrics = {k: v[step] for k, v in metrics_data.items()}
                        for callback in callbacks:
                            callback.on_step(step, step_metrics)

                final_metrics = jax.tree.map(
                    lambda x: (
                        x[-1] if hasattr(x, "__len__") and len(x.shape) > 0 else x
                    ),
                    metrics_data,
                )
                final_metrics = {
                    k: float(jnp.mean(v)) if isinstance(v, jnp.ndarray) else v
                    for k, v in final_metrics.items()
                }

                for callback in callbacks:
                    callback.on_training_end(final_metrics)

            return result

        return logged_train_fn

    return training_wrapper


def build_tqdm_progress_bar(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    **kwargs,
) -> Tuple[Callable, Callable]:
    """Build real-time tqdm progress bar for JAX computations."""
    if tqdm is None:
        # Fallback to simple print-based progress
        def _update_simple(iter_num):
            if iter_num % max(1, n // 10) == 0:
                progress = iter_num / n * 100
                print(
                    f"\r{desc or 'Progress'}: {progress:.1f}% ({iter_num}/{n})",
                    end="",
                    flush=True,
                )

        def _close_simple(result, iter_num):
            if iter_num == n - 1:
                print(f"\r{desc or 'Progress'}: 100.0% ({n}/{n}) ✓")
            return result

        return _update_simple, _close_simple

    if desc is None:
        desc = f"Training ({n:,} updates)"

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}

    if print_rate is None:
        print_rate = max(1, min(n // 20, 50))
    else:
        print_rate = max(1, min(print_rate, n))

    remainder = n % print_rate

    def _define_tqdm():
        tqdm_bars[0] = tqdm(total=n, desc=desc, unit="step", **kwargs)

    def _update_tqdm(steps):
        if 0 in tqdm_bars:
            tqdm_bars[0].update(int(steps))

    def _close_tqdm():
        if 0 in tqdm_bars:
            tqdm_bars[0].close()

    def _update_progress_bar(iter_num):
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: io_callback(_define_tqdm, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            (iter_num % print_rate == 0) & (iter_num != n - remainder) & (iter_num > 0),
            lambda _: io_callback(_update_tqdm, None, print_rate, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            iter_num == n - remainder,
            lambda _: io_callback(_update_tqdm, None, remainder, ordered=True),
            lambda _: None,
            operand=None,
        )

    def close_progress_bar(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: io_callback(_close_tqdm, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, close_progress_bar


def build_tqdm_progress_bar_with_metrics(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    metric_keys: List[str] = None,
    **kwargs,
) -> Tuple[Callable, Callable, Callable]:
    if tqdm is None:
        current_metrics = {}

        def _update_simple(iter_num):
            if iter_num % max(1, n // 10) == 0:
                progress = iter_num / n * 100
                metrics_str = ""
                if current_metrics:
                    metrics_parts = [f"{k}={v:.3f}" for k, v in current_metrics.items()]
                    metrics_str = f" | {' | '.join(metrics_parts)}"
                print(
                    f"\r{desc or 'Progress'}: {progress:.1f}% ({iter_num}/{n}){metrics_str}",
                    end="",
                    flush=True,
                )

        def _update_metrics_simple(metrics_dict):
            current_metrics.update(metrics_dict)

        def _close_simple(result, iter_num):
            if iter_num == n - 1:
                metrics_str = ""
                if current_metrics:
                    metrics_parts = [f"{k}={v:.3f}" for k, v in current_metrics.items()]
                    metrics_str = f" | {' | '.join(metrics_parts)}"
                print(f"\r{desc or 'Progress'}: 100.0% ({n}/{n}){metrics_str} ✓")
            return result

        return _update_simple, _update_metrics_simple, _close_simple

    if desc is None:
        desc = f"Training ({n:,} updates)"

    if metric_keys is None:
        metric_keys = []

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}
    current_metrics = {}

    if print_rate is None:
        print_rate = max(1, min(n // 20, 50))
    else:
        print_rate = max(1, min(print_rate, n))

    remainder = n % print_rate

    def _define_tqdm():
        tqdm_bars[0] = tqdm(total=n, desc=desc, unit="step", **kwargs)

    def _update_tqdm(steps):
        if 0 in tqdm_bars:
            tqdm_bars[0].update(int(steps))
            if current_metrics:
                filtered_metrics = {
                    k: v
                    for k, v in current_metrics.items()
                    if not metric_keys or k in metric_keys
                }
                if filtered_metrics:
                    tqdm_bars[0].set_postfix(filtered_metrics)

    def _update_metrics(metrics_dict):
        current_metrics.update(metrics_dict)
        if 0 in tqdm_bars:
            filtered_metrics = {
                k: v
                for k, v in current_metrics.items()
                if not metric_keys or k in metric_keys
            }
            if filtered_metrics:
                tqdm_bars[0].set_postfix(filtered_metrics)

    def _close_tqdm():
        if 0 in tqdm_bars:
            tqdm_bars[0].close()

    def _update_progress_bar(iter_num):
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: io_callback(_define_tqdm, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            (iter_num % print_rate == 0) & (iter_num != n - remainder) & (iter_num > 0),
            lambda _: io_callback(_update_tqdm, None, print_rate, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            iter_num == n - remainder,
            lambda _: io_callback(_update_tqdm, None, remainder, ordered=True),
            lambda _: None,
            operand=None,
        )

    def update_metrics(metrics_dict):
        for k, v in metrics_dict.items():
            if isinstance(v, jnp.ndarray):
                if v.ndim == 0:
                    metric_value = v
                else:
                    metric_value = jnp.mean(v)
            else:
                metric_value = v

            io_callback(
                lambda val, key=k: _update_metrics({key: float(val)}),
                None,
                metric_value,
                ordered=True,
            )

    def close_progress_bar(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: io_callback(_close_tqdm, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, update_metrics, close_progress_bar


def scan_with_progress(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    **tqdm_kwargs,
) -> Callable:
    """Decorator to add real-time progress bar to JAX scan operations."""
    _update_progress_bar, close_progress_bar = build_tqdm_progress_bar(
        n, print_rate, desc, **tqdm_kwargs
    )

    def _scan_progress_decorator(func):
        def wrapper_with_progress(carry, x):
            if isinstance(x, tuple):
                iter_num = x[0]
            else:
                iter_num = x

            _update_progress_bar(iter_num)

            result = func(carry, x)

            return close_progress_bar(result, iter_num)

        return wrapper_with_progress

    return _scan_progress_decorator


def fori_loop_with_progress(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    **tqdm_kwargs,
) -> Callable:
    """Decorator to add real-time progress bar to JAX fori_loop operations."""
    _update_progress_bar, close_progress_bar = build_tqdm_progress_bar(
        n, print_rate, desc, **tqdm_kwargs
    )

    def _fori_progress_decorator(func):
        def wrapper_with_progress(i, carry):
            _update_progress_bar(i)

            result = func(i, carry)

            return close_progress_bar(result, i)

        return wrapper_with_progress

    return _fori_progress_decorator


def scan_with_progress_and_metrics(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    metric_keys: List[str] = None,
    **tqdm_kwargs,
) -> Callable:
    """Decorator to add real-time progress bar with metrics to JAX scan operations."""
    _update_progress_bar, update_metrics, close_progress_bar = (
        build_tqdm_progress_bar_with_metrics(
            n, print_rate, desc, metric_keys, **tqdm_kwargs
        )
    )

    def _scan_progress_decorator(func):
        def wrapper_with_progress(carry, x):
            if isinstance(x, tuple):
                iter_num = x[0]
            else:
                iter_num = x

            _update_progress_bar(iter_num)

            result = func(carry, x)

            if isinstance(result, tuple) and len(result) == 2:
                new_carry, outputs = result
                if isinstance(outputs, tuple) and len(outputs) >= 1:
                    first_output = outputs[0]
                    if isinstance(first_output, dict):
                        update_metrics(first_output)

            return close_progress_bar(result, iter_num)

        return wrapper_with_progress

    return _scan_progress_decorator


def fori_loop_with_progress_and_metrics(
    n: int,
    print_rate: Optional[int] = None,
    desc: str = None,
    metric_keys: List[str] = None,
    **tqdm_kwargs,
) -> Callable:
    """Decorator to add real-time progress bar with metrics to JAX fori_loop operations."""
    _update_progress_bar, update_metrics, close_progress_bar = (
        build_tqdm_progress_bar_with_metrics(
            n, print_rate, desc, metric_keys, **tqdm_kwargs
        )
    )

    def _fori_progress_decorator(func):
        def wrapper_with_progress(i, carry):
            _update_progress_bar(i)

            result = func(i, carry)

            return close_progress_bar(result, i)

        return wrapper_with_progress

    return _fori_progress_decorator
