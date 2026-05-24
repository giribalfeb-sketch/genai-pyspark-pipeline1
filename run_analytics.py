"""Run PySpark sales analytics and display results in the console."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import config
from src.spark_analytics import SalesAnalytics

T = TypeVar("T")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def time_operation(name: str, operation: Callable[[], T]) -> T:
    """Run an operation, print its elapsed time, and return its result."""
    start_time = time.perf_counter()
    result = operation()
    elapsed_seconds = time.perf_counter() - start_time
    print(f"{name}: {elapsed_seconds:.2f} seconds")
    return result


def show_result(title: str, dataframe: DataFrame, rows: int = 10) -> None:
    """Print a title and display a Spark DataFrame."""
    print(f"\n{title}")
    dataframe.show(rows, truncate=False)


def format_currency(column_name: str) -> F.Column:
    """Format a numeric Spark column as US currency."""
    return F.concat(F.lit("$"), F.format_number(F.col(column_name), 2))


def format_millions(column_name: str) -> F.Column:
    """Format a numeric Spark column as compact millions when appropriate."""
    return F.when(
        F.abs(F.col(column_name)) >= 1_000_000,
        F.concat(F.lit("$"), F.format_number(F.col(column_name) / 1_000_000, 1), F.lit("M")),
    ).otherwise(format_currency(column_name))


def format_count(column_name: str) -> F.Column:
    """Format an integer Spark column with comma separators."""
    return F.format_number(F.col(column_name), 0)


def show_top_customers(dataframe: DataFrame) -> None:
    """Display top customers with formatted revenue."""
    formatted_df = dataframe.select(
        "customer_id",
        format_currency("total_revenue").alias("total_revenue"),
    )
    show_result("Top 10 Customers by Revenue", formatted_df)


def show_sales_by_category(dataframe: DataFrame) -> None:
    """Display sales by category with formatted units and revenue."""
    formatted_df = dataframe.select(
        "category",
        format_count("units_sold").alias("units_sold"),
        format_millions("total_revenue").alias("revenue"),
    )
    show_result("Sales by Category", formatted_df)


def show_monthly_trends(dataframe: DataFrame) -> None:
    """Display monthly trends with formatted revenue and growth percentage."""
    formatted_df = dataframe.select(
        "order_month",
        format_millions("total_revenue").alias("revenue"),
        format_millions("previous_month_revenue").alias("previous_month_revenue"),
        F.when(F.col("mom_growth_pct").isNull(), F.lit(""))
        .otherwise(F.concat(F.format_number(F.col("mom_growth_pct"), 2), F.lit("%")))
        .alias("mom_growth_pct"),
    )
    show_result("Monthly Trends", formatted_df, rows=24)


def main() -> int:
    """Load raw parquet data, run sales analytics, and print timed results."""
    raw_data_dir: Path = config.RAW_DATA_DIR
    analytics = SalesAnalytics()
    total_start_time = time.perf_counter()

    try:
        time_operation("Create Spark session", analytics.create_spark_session)

        customers_df = time_operation(
            "Load customers.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "customers.parquet"),
        )
        products_df = time_operation(
            "Load products.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "products.parquet"),
        )
        orders_df = time_operation(
            "Load orders.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "orders.parquet"),
        )

        print(
            "\nLoaded DataFrames: "
            f"customers={customers_df.columns}, "
            f"products={products_df.columns}, "
            f"orders={orders_df.columns}"
        )

        top_customers_df = time_operation(
            "Top customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders_df, products_df, n=10),
        )
        show_top_customers(top_customers_df)

        sales_by_category_df = time_operation(
            "Sales by category",
            lambda: analytics.sales_by_category(orders_df, products_df),
        )
        show_sales_by_category(sales_by_category_df)

        monthly_trends_df = time_operation(
            "Monthly trends",
            lambda: analytics.monthly_trends(orders_df, products_df),
        )
        show_monthly_trends(monthly_trends_df)

        total_elapsed_seconds = time.perf_counter() - total_start_time
        print(f"\nCompleted in {total_elapsed_seconds:.1f} seconds")

        return 0
    except Exception:
        logger.exception("Analytics run failed")
        return 1
    finally:
        analytics.stop()


if __name__ == "__main__":
    raise SystemExit(main())
