import duckdb
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

DB_PATH = "db/agriculture.duckdb"

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1,
    max_tokens=300,
    timeout=180
)


def query_database(sql):
    con = duckdb.connect(DB_PATH, read_only=True)

    try:
        result = con.execute(sql).fetchall()
        columns = [x[0] for x in con.description]

        return columns, result

    finally:
        con.close()


def answer_crop_yield1(question):

    sql = """
    SELECT
        COUNT(*) AS records,
        AVG(yield_tons_per_hectare) AS average_yield,
        MIN(yield_tons_per_hectare) AS minimum_yield,
        MAX(yield_tons_per_hectare) AS maximum_yield
    FROM crop_yield1
    WHERE LOWER(crop) = 'cotton'
    AND LOWER(soil_type) = 'sandy'
    """

    columns, rows = query_database(sql)

    records, average, minimum, maximum = rows[0]

    context = f"""
Dataset: crop_yield1

Question: {question}

Number of matching records: {records}
Average yield: {average:.2f} tons/hectare
Minimum yield: {minimum:.2f} tons/hectare
Maximum yield: {maximum:.2f} tons/hectare
"""

    prompt = f"""
You are AgriBot.

Answer the user's question using ONLY the supplied dataset result.

Give a SHORT answer.

Do not explain your reasoning.
Do not repeat the entire dataset.
Do not invent information.

DATASET RESULT:
{context}

USER QUESTION:
{question}

Answer in 1-3 sentences.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = input("Ask: ")

    answer = answer_crop_yield1(question)

    print("\nAgriBot:")
    print(answer)