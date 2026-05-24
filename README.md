# E-Commerce Data Pipeline

This project generates fake e-commerce data and analyzes it with PySpark to produce business insights.

## Project Purpose

- Generate synthetic customer, product, and order data for testing.
- Analyze revenue, customers, products, and order status trends using PySpark.
- Save raw CSV files in `data/raw/` and processed analytics outputs in `data/processed/`.

## Folder Structure

```text
genai-pyspark-pipeline1/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PySpark also needs a compatible Java runtime. Spark 4.1 runs on Java 17 or Java 21. If multiple Java versions are installed, set `JAVA_HOME` to your JDK 17 or JDK 21 folder before running Spark.

## Generate Fake Data

```powershell
python -m src.data_generator
```

Optional custom row counts:

```powershell
python -m src.data_generator --customers 1000 --products 100 --orders 5000
```

This creates:

- `data/raw/customers.csv`
- `data/raw/products.csv`
- `data/raw/orders.csv`

## Run PySpark Analytics

```powershell
python -m src.spark_analytics
```

This creates processed result folders in `data/processed/`:

- `revenue_by_category`
- `monthly_revenue`
- `top_customers`
- `order_status_summary`

## Run Tests

```powershell
pytest
```
