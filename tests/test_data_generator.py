"""Tests for the synthetic e-commerce data generator."""

from pathlib import Path

from src.data_generator import generate_all_data


def test_generate_all_data_creates_expected_files(tmp_path: Path) -> None:
    """Verify data generation creates the expected files and row counts."""
    customers, products, orders = generate_all_data(
        customer_count=5,
        product_count=3,
        order_count=10,
        output_dir=tmp_path,
        seed=123,
    )

    assert len(customers) == 5
    assert len(products) == 3
    assert len(orders) == 10
    assert (tmp_path / "customers.csv").exists()
    assert (tmp_path / "products.csv").exists()
    assert (tmp_path / "orders.csv").exists()
