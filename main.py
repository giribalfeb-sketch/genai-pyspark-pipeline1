"""Entry point for generating raw e-commerce data as Parquet files."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format a byte count as a readable file size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def save_parquet(dataframe: pd.DataFrame, output_path: Path) -> int:
    """Save a pandas DataFrame as a Parquet file and return the file size in bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    file_size = output_path.stat().st_size
    logger.info("Saved %s rows to %s", len(dataframe), output_path)
    return file_size


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data generation job."""
    parser = argparse.ArgumentParser(description="Generate e-commerce raw data as Parquet files.")
    parser.add_argument("--customers", type=int, default=config.DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=config.DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=config.DEFAULT_ORDER_COUNT)
    parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    """Generate customers, products, and orders and save them as Parquet files."""
    args = parse_args()
    start_time = time.perf_counter()

    try:
        logger.info("Starting synthetic data generation")
        generator = SyntheticDataGenerator(
            customer_count=args.customers,
            product_count=args.products,
            order_count=args.orders,
            seed=args.seed,
        )

        customers, products, orders = generator.generate_all()

        output_paths = {
            "customers": args.output_dir / "customers.parquet",
            "products": args.output_dir / "products.parquet",
            "orders": args.output_dir / "orders.parquet",
        }
        file_sizes = {
            "customers": save_parquet(customers, output_paths["customers"]),
            "products": save_parquet(products, output_paths["products"]),
            "orders": save_parquet(orders, output_paths["orders"]),
        }

        elapsed_seconds = time.perf_counter() - start_time

        print("\nData generation completed successfully.")
        print(f"Generation time: {elapsed_seconds:.2f} seconds")
        print("Parquet files:")
        for name, path in output_paths.items():
            print(f"- {name}: {path} ({format_file_size(file_sizes[name])})")

        return 0
    except Exception:
        logger.exception("Data generation failed")
        print("Data generation failed. Check the log output above for details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
