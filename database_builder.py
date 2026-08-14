import os
import duckdb

DATA_DIR = "processed_data"
DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "agriculture.duckdb")

os.makedirs(DB_DIR, exist_ok=True)

DATASETS = [
    "all_india_dataset_final.csv",
    "crop_yield1.csv",
    "crop_yield2.csv",
    "Custom_Crops_yield_Historical_Dataset.csv",
    "Indian_crop_production_yield_dataset.csv",
    "state_soil_data.csv",
    "state_weather_data_1997_2020.csv",
    "synthetic_crop_yield_data.csv"
]

con = duckdb.connect(DB_PATH)

print("=" * 70)
print("BUILDING AGRICULTURE DUCKDB DATABASE")
print("=" * 70)

for filename in DATASETS:
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        print(f"SKIPPED: {filename}")
        continue

    table_name = os.path.splitext(filename)[0]
    table_name = table_name.lower().replace("-", "_").replace(" ", "_")

    print(f"\nLoading: {filename}")

    con.execute(f"""
        CREATE OR REPLACE TABLE "{table_name}" AS
        SELECT *
        FROM read_csv_auto(
            '{path.replace(chr(92), "/")}',
            header=true,
            ignore_errors=false
        )
    """)

    count = con.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    ).fetchone()[0]

    print(f"Table: {table_name}")
    print(f"Rows: {count:,}")

print("\n" + "=" * 70)
print("CREATING INDEXES")
print("=" * 70)

print("\nCreating database summary...")

tables = con.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'main'
    ORDER BY table_name
""").fetchall()

total_rows = 0

for (table,) in tables:
    count = con.execute(
        f'SELECT COUNT(*) FROM "{table}"'
    ).fetchone()[0]

    total_rows += count
    print(f"{table}: {count:,} rows")

print("\n" + "=" * 70)
print("DATABASE BUILD COMPLETED")
print("=" * 70)
print(f"Total rows: {total_rows:,}")
print(f"Database: {DB_PATH}")

con.close()