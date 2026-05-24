"""Compare Pandas and PySpark performance for a sales aggregation workload."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config
from src.spark_analytics import SalesAnalytics

T = TypeVar("T")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkResult:
    """Store timed benchmark results for one processing engine."""

    engine: str
    load_seconds: float
    join_revenue_seconds: float
    aggregate_seconds: float
    total_seconds: float


def time_operation(operation: Callable[[], T]) -> tuple[T, float]:
    """Run an operation and return its result with elapsed wall-clock seconds."""
    start_time = time.perf_counter()
    result = operation()
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds


def run_pandas_benchmark(orders_path: Path, products_path: Path) -> tuple[BenchmarkResult, pd.DataFrame]:
    """Run the join, revenue calculation, and top-customer aggregation with Pandas."""
    total_start = time.perf_counter()

    def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        orders = pd.read_parquet(orders_path)
        products = pd.read_parquet(products_path)
        return orders, products

    (orders_df, products_df), load_seconds = time_operation(load_data)

    def join_and_calculate_revenue() -> pd.DataFrame:
        joined = orders_df.merge(products_df[["product_id", "price"]], on="product_id", how="inner")
        joined["revenue"] = joined["quantity"] * joined["price"]
        return joined

    joined_df, join_revenue_seconds = time_operation(join_and_calculate_revenue)

    def aggregate_top_customers() -> pd.DataFrame:
        return (
            joined_df.groupby("customer_id", as_index=False)["revenue"]
            .sum()
            .sort_values("revenue", ascending=False)
            .head(10)
        )

    top_customers_df, aggregate_seconds = time_operation(aggregate_top_customers)
    total_seconds = time.perf_counter() - total_start

    result = BenchmarkResult(
        engine="Pandas",
        load_seconds=load_seconds,
        join_revenue_seconds=join_revenue_seconds,
        aggregate_seconds=aggregate_seconds,
        total_seconds=total_seconds,
    )
    return result, top_customers_df


def create_spark_session() -> SparkSession:
    """Create a Spark session using the project analytics configuration."""
    return SalesAnalytics(app_name="PandasVsPySparkBenchmark").create_spark_session()


def run_pyspark_benchmark(
    spark: SparkSession,
    orders_path: Path,
    products_path: Path,
) -> tuple[BenchmarkResult, DataFrame]:
    """Run the join, revenue calculation, and top-customer aggregation with PySpark."""
    total_start = time.perf_counter()

    def load_data() -> tuple[DataFrame, DataFrame]:
        orders = spark.read.parquet(str(orders_path))
        products = spark.read.parquet(str(products_path))
        orders.count()
        products.count()
        return orders, products

    (orders_df, products_df), load_seconds = time_operation(load_data)

    def join_and_calculate_revenue() -> DataFrame:
        joined = (
            orders_df.alias("orders")
            .join(products_df.select("product_id", "price").alias("products"), on="product_id", how="inner")
            .withColumn("revenue", F.col("orders.quantity") * F.col("products.price"))
            .cache()
        )
        joined.count()
        return joined

    joined_df, join_revenue_seconds = time_operation(join_and_calculate_revenue)

    def aggregate_top_customers() -> DataFrame:
        top_customers = (
            joined_df.groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("revenue"))
            .orderBy(F.desc("revenue"))
            .limit(10)
        )
        top_customers.count()
        return top_customers

    top_customers_df, aggregate_seconds = time_operation(aggregate_top_customers)
    total_seconds = time.perf_counter() - total_start

    result = BenchmarkResult(
        engine="PySpark",
        load_seconds=load_seconds,
        join_revenue_seconds=join_revenue_seconds,
        aggregate_seconds=aggregate_seconds,
        total_seconds=total_seconds,
    )
    return result, top_customers_df


def print_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a comparison table for Pandas and PySpark timings."""
    headers = ["Engine", "Load s", "Join + Revenue s", "Aggregate s", "Total s"]
    rows = [
        [
            result.engine,
            f"{result.load_seconds:.2f}",
            f"{result.join_revenue_seconds:.2f}",
            f"{result.aggregate_seconds:.2f}",
            f"{result.total_seconds:.2f}",
        ]
        for result in results
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)

    print("\nPandas vs PySpark Performance Comparison")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    """Run the Pandas and PySpark performance comparison."""
    orders_path = config.RAW_DATA_DIR / "orders.parquet"
    products_path = config.RAW_DATA_DIR / "products.parquet"
    spark: SparkSession | None = None

    try:
        pandas_result, pandas_top_customers = run_pandas_benchmark(orders_path, products_path)

        spark = create_spark_session()
        pyspark_result, pyspark_top_customers = run_pyspark_benchmark(
            spark,
            orders_path,
            products_path,
        )

        print_comparison_table([pandas_result, pyspark_result])

        print("\nPandas Top 10 Customers")
        print(pandas_top_customers.to_string(index=False))

        print("\nPySpark Top 10 Customers")
        pyspark_top_customers.show(truncate=False)

        return 0
    except Exception:
        logger.exception("Pandas vs PySpark comparison failed")
        return 1
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    raise SystemExit(main())
