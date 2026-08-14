import os
import duckdb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_PATH = "db/agriculture.duckdb"
CHROMA_PATH = "db/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

con = duckdb.connect(DB_PATH, read_only=True)

vectorstore = Chroma(
    collection_name="agriculture",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

documents = []


def add_table_chunks(table, columns, group_columns, chunk_size=100):
    print(f"\nProcessing: {table}")

    select_columns = ", ".join(f'"{c}"' for c in columns)
    group_select = ", ".join(f'"{c}"' for c in group_columns)

    groups = con.execute(f"""
        SELECT DISTINCT {group_select}
        FROM "{table}"
    """).fetchall()

    print(f"Groups found: {len(groups)}")

    for group in groups:
        conditions = []

        for column, value in zip(group_columns, group):
            if value is None:
                conditions.append(f'"{column}" IS NULL')
            elif isinstance(value, (int, float)):
                conditions.append(f'"{column}" = {value}')
            else:
                safe_value = str(value).replace("'", "''")
                conditions.append(f'"{column}" = \'{safe_value}\'')

        where_clause = " AND ".join(conditions)

        rows = con.execute(f"""
            SELECT {select_columns}
            FROM "{table}"
            WHERE {where_clause}
        """).fetchall()

        for start in range(0, len(rows), chunk_size):
            chunk = rows[start:start + chunk_size]

            text_parts = []

            for row in chunk:
                record = []

                for column, value in zip(columns, row):
                    record.append(f"{column}: {value}")

                text_parts.append(" | ".join(record))

            text = "\n".join(text_parts)

            metadata = {
                "source": table,
                "group": " | ".join(
                    f"{column}={value}"
                    for column, value in zip(group_columns, group)
                )
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )
add_table_chunks(
    "crop_yield1",
    [
        "region",
        "soil_type",
        "crop",
        "rainfall_mm",
        "temperature_celsius",
        "fertilizer_used",
        "irrigation_used",
        "weather_condition",
        "days_to_harvest",
        "yield_tons_per_hectare"
    ],
    ["region", "crop", "soil_type"],
    1000
)

add_table_chunks(
    "all_india_dataset_final",
    [
        "state_names",
        "district_names",
        "crop_year",
        "season_names",
        "crop_names",
        "area",
        "temperature",
        "wind_speed",
        "precipitation",
        "humidity",
        "soil_type",
        "N",
        "P",
        "K",
        "production",
        "pressure"
    ],
    ["state_names", "crop_names"],
    100
)

add_table_chunks(
    "crop_yield2",
    [
        "crop",
        "year",
        "season",
        "state",
        "area",
        "production",
        "fertilizer",
        "pesticide",
        "yield"
    ],
    ["state", "crop"],
    100
)

add_table_chunks(
    "custom_crops_yield_historical_dataset",
    [
        "state_name",
        "dist_name",
        "year",
        "crop",
        "area_ha",
        "yield_kg_per_ha",
        "n_req_kg_per_ha",
        "p_req_kg_per_ha",
        "k_req_kg_per_ha",
        "total_n_kg",
        "total_p_kg",
        "total_k_kg",
        "temperature_c",
        "humidity_%",
        "ph",
        "rainfall_mm",
        "wind_speed_m_s",
        "solar_radiation_mj_m2_day"
    ],
    ["state_name", "crop"],
    100
)

add_table_chunks(
    "indian_crop_production_yield_dataset",
    [
        "State_Name",
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
        "Production",
        "yield"
    ],
    ["State_Name", "Crop"],
    100
)

add_table_chunks(
    "state_soil_data",
    [
        "state",
        "N",
        "P",
        "K",
        "pH"
    ],
    ["state"],
    100
)

add_table_chunks(
    "state_weather_data_1997_2020",
    [
        "state",
        "year",
        "avg_temp_c",
        "total_rainfall_mm",
        "avg_humidity_percent"
    ],
    ["state"],
    100
)

add_table_chunks(
    "synthetic_crop_yield_data",
    [
        "Crop_Type",
        "Precipitation",
        "Temperature",
        "Soil_Quality",
        "Fertilizer_Usage",
        "Pesticide_Usage",
        "Irrigation_Water",
        "Humidity",
        "Season",
        "Crop_Yield"
    ],
    ["Crop_Type", "Season"],
    100
)


print("\n" + "=" * 70)
print(f"TOTAL CHUNKS CREATED: {len(documents):,}")
print("=" * 70)

batch_size = 64

for start in range(0, len(documents), batch_size):
    batch = documents[start:start + batch_size]

    vectorstore.add_documents(batch)

    done = min(start + batch_size, len(documents))

    print(f"Embedded {done:,}/{len(documents):,}")

print("\n" + "=" * 70)
print("VECTOR BUILD COMPLETED")
print("=" * 70)
print(f"Chroma database: {CHROMA_PATH}")

con.close()