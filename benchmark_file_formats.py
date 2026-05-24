"""Benchmark common data file formats with timing, memory, and energy metrics."""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.orc as orc

CPU_TDP_WATTS = 65
DEFAULT_ROW_COUNT = 500_000
OUTPUT_DIR = Path("data") / "format_benchmarks"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Store benchmark metrics for one file format."""

    format_name: str
    file_path: Path
    file_size_mb: float
    write_time_seconds: float
    read_time_seconds: float
    peak_memory_mb: float
    cpu_time_seconds: float
    energy_wh: float


def create_benchmark_dataframe(row_count: int = DEFAULT_ROW_COUNT) -> pd.DataFrame:
    """Create a deterministic e-commerce-style DataFrame for file format benchmarking."""
    rng = np.random.default_rng(42)
    ids = np.arange(1, row_count + 1)
    categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])

    dataframe = pd.DataFrame(
        {
            "id": ids,
            "name": pd.Series(ids).map(lambda value: f"Customer {value:06d}"),
            "email": pd.Series(ids).map(lambda value: f"customer{value:06d}@example.com"),
            "amount": rng.gamma(shape=2.0, scale=75.0, size=row_count).round(2),
            "date": pd.date_range("2024-01-01", periods=row_count, freq="min").strftime("%Y-%m-%d"),
            "category": rng.choice(categories, size=row_count),
        }
    )
    logger.info("Created benchmark DataFrame with shape %s", dataframe.shape)
    return dataframe


def get_file_size_mb(path: Path) -> float:
    """Return the size of a file in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def estimate_energy_wh(cpu_time_seconds: float, cpu_tdp_watts: int = CPU_TDP_WATTS) -> float:
    """Estimate energy consumption in watt-hours from CPU time and CPU TDP."""
    return cpu_time_seconds * cpu_tdp_watts / 3600


def measure_operation(operation: Callable[[], object]) -> tuple[float, float, float]:
    """Measure wall time, CPU time, and peak traced memory for one operation."""
    gc.collect()
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    try:
        operation()
        cpu_time_seconds = time.process_time() - cpu_start
        wall_time_seconds = time.perf_counter() - wall_start
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return wall_time_seconds, cpu_time_seconds, peak_memory_bytes / (1024 * 1024)


def write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""
    dataframe.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def write_xlsx(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to XLSX."""
    dataframe.to_excel(path, index=False, engine="openpyxl")


def read_xlsx(path: Path) -> pd.DataFrame:
    """Read an XLSX file into a DataFrame."""
    return pd.read_excel(path, engine="openpyxl")


def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Parquet using fastparquet."""
    dataframe.to_parquet(path, index=False, engine="fastparquet", compression="snappy")


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame using fastparquet."""
    return pd.read_parquet(path, engine="fastparquet")


def write_orc(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to ORC using PyArrow."""
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    with path.open("wb") as output_file:
        orc.write_table(table, output_file)


def read_orc(path: Path) -> pd.DataFrame:
    """Read an ORC file into a DataFrame using PyArrow."""
    with path.open("rb") as input_file:
        return orc.ORCFile(input_file).read().to_pandas()


def write_feather(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to Feather using PyArrow."""
    feather.write_feather(dataframe, path, compression="lz4")


def read_feather(path: Path) -> pd.DataFrame:
    """Read a Feather file into a DataFrame using PyArrow."""
    return feather.read_feather(path)


def benchmark_format(
    dataframe: pd.DataFrame,
    format_name: str,
    file_path: Path,
    writer: Callable[[pd.DataFrame, Path], None],
    reader: Callable[[Path], pd.DataFrame],
) -> BenchmarkResult:
    """Benchmark write/read performance and resource metrics for a single format."""
    logger.info("Benchmarking %s", format_name)

    write_time, write_cpu_time, write_memory = measure_operation(lambda: writer(dataframe, file_path))
    file_size_mb = get_file_size_mb(file_path)

    read_time, read_cpu_time, read_memory = measure_operation(lambda: reader(file_path))
    total_cpu_time = write_cpu_time + read_cpu_time

    return BenchmarkResult(
        format_name=format_name,
        file_path=file_path,
        file_size_mb=file_size_mb,
        write_time_seconds=write_time,
        read_time_seconds=read_time,
        peak_memory_mb=max(write_memory, read_memory),
        cpu_time_seconds=total_cpu_time,
        energy_wh=estimate_energy_wh(total_cpu_time),
    )


def percent_savings(baseline: float, value: float) -> float:
    """Calculate percentage savings compared with a baseline value."""
    if baseline == 0:
        return 0.0
    return ((baseline - value) / baseline) * 100


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark metrics and percentage savings versus the CSV baseline."""
    csv_result = next(result for result in results if result.format_name == "CSV")

    headers = [
        "Format",
        "Size MB",
        "Write s",
        "Read s",
        "Peak Mem MB",
        "CPU s",
        "Energy Wh",
        "Size vs CSV",
        "Write vs CSV",
        "Read vs CSV",
        "Energy vs CSV",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.format_name,
                f"{result.file_size_mb:.2f}",
                f"{result.write_time_seconds:.2f}",
                f"{result.read_time_seconds:.2f}",
                f"{result.peak_memory_mb:.2f}",
                f"{result.cpu_time_seconds:.2f}",
                f"{result.energy_wh:.4f}",
                f"{percent_savings(csv_result.file_size_mb, result.file_size_mb):.1f}%",
                f"{percent_savings(csv_result.write_time_seconds, result.write_time_seconds):.1f}%",
                f"{percent_savings(csv_result.read_time_seconds, result.read_time_seconds):.1f}%",
                f"{percent_savings(csv_result.energy_wh, result.energy_wh):.1f}%",
            ]
        )

    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)

    print("\nFile Format Benchmark Results")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def run_benchmark(row_count: int = DEFAULT_ROW_COUNT, output_dir: Path = OUTPUT_DIR) -> list[BenchmarkResult]:
    """Run the benchmark for CSV, XLSX, Parquet, ORC, and Feather formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = create_benchmark_dataframe(row_count)

    benchmarks = [
        ("CSV", output_dir / "benchmark.csv", write_csv, read_csv),
        ("XLSX", output_dir / "benchmark.xlsx", write_xlsx, read_xlsx),
        ("Parquet", output_dir / "benchmark.parquet", write_parquet, read_parquet),
        ("ORC", output_dir / "benchmark.orc", write_orc, read_orc),
        ("Feather", output_dir / "benchmark.feather", write_feather, read_feather),
    ]

    results = [
        benchmark_format(dataframe, format_name, path, writer, reader)
        for format_name, path, writer, reader in benchmarks
    ]
    print_comparison_table(results)
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark CSV, XLSX, Parquet, ORC, and Feather.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing benchmark output files before running.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the file format benchmark script."""
    args = parse_args()
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    try:
        run_benchmark(row_count=args.rows, output_dir=args.output_dir)
        return 0
    except Exception:
        logger.exception("Benchmark failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
