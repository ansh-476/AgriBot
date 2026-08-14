from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import re

from retrieval_pipeline import semantic_search

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "AgriBot API running"}


def split_questions(text):
    parts = re.split(r'\?', text)
    questions = []

    for part in parts:
        part = part.strip()
        if part:
            questions.append(part + "?")

    return questions


@app.post("/api/chat")
def chat(request: ChatRequest):

    questions = split_questions(request.query)

    all_answers = []

    for question in questions:

        results = semantic_search(question, k=4)

        if not results:
            all_answers.append(
                f"**{question}**\nI could not find enough information in the provided dataset."
            )
            continue

        context = "\n\n".join(
            result["content"] for result in results
        )

        prompt = f"""
You are AgriBot, an agriculture dataset assistant.

Answer ONLY using the dataset context.

Rules:
- Answer the question directly.
- Give 2-3 sentences.
- Use exact numerical values from the dataset.
- Do not use outside knowledge.
- Do not guess.
- Do not invent information.
- Do not show reasoning.
- If the dataset does not contain enough information, say:
"I could not find enough information in the provided dataset."

DATASET:
{context}

QUESTION:
{question}

Return only the final answer.
"""

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 250,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

        try:
            response = requests.post(
                NVIDIA_URL,
                headers=headers,
                json=payload,
                timeout=(10, 120)
            )

            response.raise_for_status()

            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()

            all_answers.append(
                f"**{question}**\n{answer}"
            )

        except Exception:
            all_answers.append(
                f"**{question}**\nI could not retrieve an answer from the provided dataset."
            )

    return {
        "reply": "\n\n".join(all_answers)
    }
