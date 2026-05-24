"""PySpark sales analytics for synthetic e-commerce data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Run sales analytics over generated e-commerce parquet data."""

    def __init__(self, app_name: str = config.SPARK_APP_NAME) -> None:
        """Initialize the analytics class with a Spark application name."""
        self.app_name = app_name
        self.spark: SparkSession | None = None

    def create_spark_session(self) -> SparkSession:
        """Create a local SparkSession with performance-oriented settings."""
        logger.info("Creating Spark session: %s", self.app_name)
        self.spark = (
            SparkSession.builder.appName(self.app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a parquet file or parquet directory into a Spark DataFrame."""
        if self.spark is None:
            self.create_spark_session()

        parquet_path = Path(path)
        logger.info("Loading parquet data from %s", parquet_path)
        return self.spark.read.parquet(str(parquet_path))

    def _orders_with_revenue(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders with products and calculate order-level revenue."""
        return (
            orders_df.alias("orders")
            .join(products_df.alias("products"), on="product_id", how="inner")
            .withColumn("revenue", F.col("orders.quantity") * F.col("products.price"))
        )

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top N customers ranked by total revenue."""
        logger.info("Calculating top %s customers by revenue", n)
        enriched_orders = self._orders_with_revenue(orders_df, products_df)

        return (
            enriched_orders.groupBy("customer_id")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.countDistinct("order_id").alias("order_count"),
                F.sum("quantity").alias("units_purchased"),
            )
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return total revenue and units sold grouped by product category."""
        logger.info("Calculating sales by category")
        enriched_orders = self._orders_with_revenue(orders_df, products_df)

        return (
            enriched_orders.groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
                F.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month revenue growth percentage."""
        logger.info("Calculating monthly revenue trends")
        enriched_orders = self._orders_with_revenue(orders_df, products_df)
        monthly_revenue = (
            enriched_orders.withColumn("order_month", F.date_format(F.col("order_date"), "yyyy-MM"))
            .groupBy("order_month")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
        )

        month_window = Window.orderBy("order_month")
        return (
            monthly_revenue.withColumn(
                "previous_month_revenue",
                F.lag("total_revenue").over(month_window),
            )
            .withColumn(
                "mom_growth_pct",
                F.when(
                    F.col("previous_month_revenue").isNull()
                    | (F.col("previous_month_revenue") == 0),
                    F.lit(None),
                ).otherwise(
                    F.round(
                        ((F.col("total_revenue") - F.col("previous_month_revenue"))
                         / F.col("previous_month_revenue"))
                        * 100,
                        2,
                    )
                ),
            )
            .orderBy("order_month")
        )

    def stop(self) -> None:
        """Stop the active Spark session if one exists."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            logger.info("Spark session stopped")


def write_dataframe(dataframe: DataFrame, output_path: Path) -> None:
    """Write a Spark DataFrame to a parquet output directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing analytics output to %s", output_path)
    dataframe.write.mode("overwrite").parquet(str(output_path))


def run_analytics(
    raw_data_dir: Path = config.RAW_DATA_DIR,
    output_dir: Path = config.PROCESSED_DATA_DIR,
) -> None:
    """Load raw parquet files, run analytics, and save parquet result datasets."""
    analytics = SalesAnalytics()

    try:
        analytics.create_spark_session()
        orders_df = analytics.load_parquet(raw_data_dir / "orders.parquet")
        products_df = analytics.load_parquet(raw_data_dir / "products.parquet")

        write_dataframe(
            analytics.top_customers_by_revenue(orders_df, products_df),
            output_dir / "top_customers_by_revenue",
        )
        write_dataframe(
            analytics.sales_by_category(orders_df, products_df),
            output_dir / "sales_by_category",
        )
        write_dataframe(
            analytics.monthly_trends(orders_df, products_df),
            output_dir / "monthly_trends",
        )
        logger.info("Sales analytics completed successfully")
    finally:
        analytics.stop()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running sales analytics."""
    parser = argparse.ArgumentParser(description="Run PySpark sales analytics over parquet data.")
    parser.add_argument("--raw-data-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the PySpark sales analytics command-line workflow."""
    args = parse_args()
    run_analytics(raw_data_dir=args.raw_data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
