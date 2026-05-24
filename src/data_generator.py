"""Generate synthetic e-commerce data for local analytics development."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import (
    CUSTOMERS_FILE,
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    ORDERS_FILE,
    PRODUCTS_FILE,
    RANDOM_SEED,
    RAW_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SyntheticDataGenerator:
    """Generate synthetic customer, product, and order data for e-commerce analytics."""

    customer_count: int = 100_000
    product_count: int = 10_000
    order_count: int = 1_000_000
    seed: int = RANDOM_SEED
    faker: Faker = field(default_factory=Faker)

    PRODUCT_CATEGORIES: ClassVar[list[str]] = [
        "Electronics",
        "Clothing",
        "Home",
        "Sports",
        "Books",
    ]

    def __post_init__(self) -> None:
        """Initialize deterministic random generators."""
        Faker.seed(self.seed)
        self.faker.seed_instance(self.seed)
        self.rng = np.random.default_rng(self.seed)
        logger.info(
            "Initialized generator with %s customers, %s products, and %s orders",
            self.customer_count,
            self.product_count,
            self.order_count,
        )

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer records with Faker names, emails, and normally distributed ages."""
        logger.info("Generating %s customers", self.customer_count)
        ages = self.rng.normal(loc=35, scale=12, size=self.customer_count)
        ages = np.clip(np.rint(ages), 18, 85).astype(int)
        registration_offsets = self.rng.integers(0, 365 * 5, size=self.customer_count)
        today = date.today()

        customers: list[dict[str, object]] = []
        for index in tqdm(range(self.customer_count), desc="Generating customers"):
            customers.append(
                {
                    "customer_id": index + 1,
                    "name": self.faker.name(),
                    "email": self.faker.unique.email(),
                    "age": int(ages[index]),
                    "city": self.faker.city(),
                    "country": self.faker.country(),
                    "registration_date": today - timedelta(days=int(registration_offsets[index])),
                }
            )

        dataframe = pd.DataFrame(customers)
        logger.info("Generated customers DataFrame with shape %s", dataframe.shape)
        return dataframe

    def generate_products(self) -> pd.DataFrame:
        """Generate product records with categories, prices, stock counts, and ratings."""
        logger.info("Generating %s products", self.product_count)
        categories = self.rng.choice(self.PRODUCT_CATEGORIES, size=self.product_count)
        prices = self.rng.uniform(10, 500, size=self.product_count).round(2)
        stock = self.rng.integers(0, 2_000, size=self.product_count)
        ratings = self.rng.uniform(1, 5, size=self.product_count).round(2)

        products: list[dict[str, object]] = []
        for index in tqdm(range(self.product_count), desc="Generating products"):
            category = str(categories[index])
            products.append(
                {
                    "product_id": index + 1,
                    "name": f"{category} Product {index + 1}",
                    "category": category,
                    "price": float(prices[index]),
                    "stock": int(stock[index]),
                    "rating": float(ratings[index]),
                }
            )

        dataframe = pd.DataFrame(products)
        logger.info("Generated products DataFrame with shape %s", dataframe.shape)
        return dataframe

    def _generate_pareto_customer_ids(self) -> np.ndarray:
        """Generate customer IDs so 20 percent of customers make 80 percent of orders."""
        logger.info("Generating Pareto-distributed customer IDs for orders")
        customer_ids = np.arange(1, self.customer_count + 1)
        high_value_count = max(1, int(self.customer_count * 0.20))

        high_value_customers = customer_ids[:high_value_count]
        regular_customers = customer_ids[high_value_count:]

        high_value_order_count = int(self.order_count * 0.80)
        regular_order_count = self.order_count - high_value_order_count

        high_value_weights = self.rng.pareto(a=1.5, size=high_value_count) + 1
        high_value_weights = high_value_weights / high_value_weights.sum()

        high_value_ids = self.rng.choice(
            high_value_customers,
            size=high_value_order_count,
            replace=True,
            p=high_value_weights,
        )

        if len(regular_customers) == 0:
            return high_value_ids

        regular_weights = self.rng.pareto(a=3.0, size=len(regular_customers)) + 1
        regular_weights = regular_weights / regular_weights.sum()
        regular_ids = self.rng.choice(
            regular_customers,
            size=regular_order_count,
            replace=True,
            p=regular_weights,
        )

        customer_order_ids = np.concatenate([high_value_ids, regular_ids])
        self.rng.shuffle(customer_order_ids)
        return customer_order_ids

    def generate_orders(self) -> pd.DataFrame:
        """Generate order records linked to customers and products."""
        logger.info("Generating %s orders", self.order_count)

        progress = tqdm(total=5, desc="Generating orders")
        customer_ids = self._generate_pareto_customer_ids()
        progress.update(1)

        product_ids = self.rng.integers(1, self.product_count + 1, size=self.order_count)
        progress.update(1)

        quantities = self.rng.integers(1, 11, size=self.order_count)
        progress.update(1)

        order_offsets = self.rng.integers(0, 365 * 2, size=self.order_count)
        today = date.today()
        order_dates = [today - timedelta(days=int(offset)) for offset in order_offsets]
        progress.update(1)

        dataframe = pd.DataFrame(
            {
                "order_id": np.arange(1, self.order_count + 1),
                "customer_id": customer_ids,
                "product_id": product_ids,
                "quantity": quantities,
                "order_date": order_dates,
            }
        )
        progress.update(1)
        progress.close()

        logger.info("Generated orders DataFrame with shape %s", dataframe.shape)
        return dataframe

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders as pandas DataFrames."""
        customers = self.generate_customers()
        products = self.generate_products()
        orders = self.generate_orders()
        return customers, products, orders

    def save_to_csv(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        orders: pd.DataFrame,
        output_dir: Path = RAW_DATA_DIR,
    ) -> None:
        """Save generated DataFrames to CSV files in the output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        customers.to_csv(output_dir / CUSTOMERS_FILE.name, index=False)
        products.to_csv(output_dir / PRODUCTS_FILE.name, index=False)
        orders.to_csv(output_dir / ORDERS_FILE.name, index=False)
        logger.info("Saved generated data files to %s", output_dir)


def generate_all_data(
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    output_dir: Path = RAW_DATA_DIR,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate all synthetic data, save it to CSV, and return the DataFrames."""
    generator = SyntheticDataGenerator(
        customer_count=customer_count,
        product_count=product_count,
        order_count=order_count,
        seed=seed,
    )
    customers, products, orders = generator.generate_all()
    generator.save_to_csv(customers, products, orders, output_dir)
    return customers, products, orders


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data generation."""
    parser = argparse.ArgumentParser(description="Generate fake e-commerce CSV data.")
    parser.add_argument("--customers", type=int, default=100_000)
    parser.add_argument("--products", type=int, default=10_000)
    parser.add_argument("--orders", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the synthetic data generator from the command line."""
    args = parse_args()
    generate_all_data(
        customer_count=args.customers,
        product_count=args.products,
        order_count=args.orders,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
