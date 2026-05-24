"""Quick benchmark for CSV, Parquet, and Feather file formats."""

from __future__ import annotations

import os
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
ROW_COUNT = 500_000
CPU_TDP_WATTS = 65
OUTPUT_DIR = Path("data") / "quick_benchmarks"


def create_test_data(row_count: int = ROW_COUNT) -> pd.DataFrame:
    """Create a synthetic DataFrame for file format benchmarking."""
    return pd.DataFrame(
        {
            "id": range(row_count),
            "name": [fake.name() for _ in range(row_count)],
            "email": [fake.email() for _ in range(row_count)],
            "amount": np.random.uniform(10, 1000, row_count).round(2),
            "date": pd.date_range("2024-01-01", periods=row_count, freq="s"),
            "category": np.random.choice(["A", "B", "C", "D"], row_count),
        }
    )


def benchmark(
    name: str,
    write_fn: Callable[[Path], None],
    read_fn: Callable[[Path], pd.DataFrame],
    path: Path,
) -> dict[str, float | str]:
    """Measure file size, write time, read time, memory, CPU time, and energy usage."""
    tracemalloc.start()
    cpu_start = time.process_time()

    write_start = time.time()
    write_fn(path)
    write_time = time.time() - write_start

    read_start = time.time()
    read_fn(path)
    read_time = time.time() - read_start

    cpu_time = time.process_time() - cpu_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size_mb = os.path.getsize(path) / 1024**2
    memory_mb = peak_memory / 1024**2
    energy_wh = cpu_time * CPU_TDP_WATTS / 3600

    return {
        "Format": name,
        "Size_MB": round(size_mb, 2),
        "Write_s": round(write_time, 2),
        "Read_s": round(read_time, 2),
        "CPU_s": round(cpu_time, 2),
        "Memory_MB": round(memory_mb, 2),
        "Energy_Wh": round(energy_wh, 4),
    }


def main() -> None:
    """Run the benchmark and print a comparison table."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = create_test_data()

    results = [
        benchmark(
            "CSV",
            lambda path: df.to_csv(path, index=False),
            pd.read_csv,
            OUTPUT_DIR / "test.csv",
        ),
        benchmark(
            "Parquet",
            lambda path: df.to_parquet(path, index=False),
            pd.read_parquet,
            OUTPUT_DIR / "test.parquet",
        ),
        benchmark(
            "Feather",
            lambda path: df.to_feather(path),
            pd.read_feather,
            OUTPUT_DIR / "test.feather",
        ),
    ]

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
