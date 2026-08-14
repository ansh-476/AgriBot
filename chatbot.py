import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=api_key,
    temperature=0.1,
    max_tokens=1024
)


SYSTEM_PROMPT = """
You are an agriculture dataset assistant.

You MUST answer using ONLY the agriculture dataset context
provided to you.

Do not use outside knowledge.

Do not invent facts, numbers, crops, locations, weather values,
soil values, yields, production values, or recommendations.

If the provided dataset context does not contain enough
information to answer the question, say:

"I could not find this information in the provided dataset."

When numerical information is available, use the exact values
from the dataset.

Clearly distinguish between different datasets when necessary.
"""


def generate_answer(question, context):
    prompt = f"""
{SYSTEM_PROMPT}

USER QUESTION:
{question}

DATASET CONTEXT:
{context}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content