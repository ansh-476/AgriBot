import warnings
import os
import logging

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)


PERSIST_DIRECTORY = "db/chroma"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def main():

    print("⚡ Loading Agriculture Vector Database...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = Chroma(
        collection_name="agriculture",
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 3}
    )

    print("🤖 Loading NVIDIA Nemotron...")

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY not found in .env"
        )

    llm = ChatNVIDIA(
        model=NVIDIA_MODEL,
        api_key=api_key,
        temperature=0.1,
        max_tokens=400,
        timeout=180
    )

    contextualize_q_system_prompt = """
You are an agriculture dataset assistant.

Given the chat history and the latest user question,
rewrite the latest question into a standalone question.

Do not answer the question.

Only rewrite the question when the previous conversation
is necessary to understand it.

If the question is already standalone, return it unchanged.
"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_system_prompt = """
You are an agriculture dataset chatbot.

You MUST answer the user's question ONLY using the
provided agriculture dataset context.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not guess missing values.
4. Do not create facts that are not present in the context.
5. Use exact numerical values when they are available.
6. Give a concise and direct answer.
7. Do not dump all retrieved dataset rows into the answer.
8. Summarize the relevant information instead.
9. If the context does not contain enough information,
   say:

"I could not find this information in the provided dataset."

10. Do not mention these instructions in your answer.

Dataset Context:
{context}
"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain
    )

    print("\n" + "=" * 70)
    print("AGRICULTURE AI CHATBOT")
    print("=" * 70)
    print("Model:", NVIDIA_MODEL)
    print("Type 'exit' to quit.")
    print("=" * 70)

    chat_history = []

    while True:

        user_query = input("\n🗣️ Ask your question: ").strip()

        if user_query.lower() in [
            "exit",
            "quit",
            "q"
        ]:
            print("Exiting chat.")
            break

        if not user_query:
            continue

        print("\n🔎 Searching agriculture dataset...")

        try:

            result = retrieval_chain.invoke({
                "input": user_query,
                "chat_history": chat_history
            })

            print("\n🌱 AI Response:")
            print("-" * 50)
            print(result["answer"])

            print("\n📚 Dataset Sources:")
            
            unique_sources = set()

            for doc in result["context"]:

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                group = doc.metadata.get(
                    "group",
                    ""
                )

                unique_sources.add(
                    f"{source} | {group}"
                )

            for i, source in enumerate(
                unique_sources,
                1
            ):
                print(f"{i}. {source}")

            print("-" * 50)

            chat_history.append(
                HumanMessage(
                    content=user_query
                )
            )

            chat_history.append(
                AIMessage(
                    content=result["answer"]
                )
            )

        except Exception as e:

            print("\n❌ Error:")
            print(e)


if __name__ == "__main__":
    main()