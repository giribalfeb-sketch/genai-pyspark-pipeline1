"""Project configuration values for the e-commerce data pipeline."""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

CUSTOMERS_FILE: Path = RAW_DATA_DIR / "customers.csv"
PRODUCTS_FILE: Path = RAW_DATA_DIR / "products.csv"
ORDERS_FILE: Path = RAW_DATA_DIR / "orders.csv"

DEFAULT_CUSTOMER_COUNT: int = 100_000
DEFAULT_PRODUCT_COUNT: int = 10_000
DEFAULT_ORDER_COUNT: int = 1_000_000
RANDOM_SEED: int = 42

SPARK_APP_NAME: str = "EcommerceDataPipeline"
