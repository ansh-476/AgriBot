from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "db/chroma"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    collection_name="agriculture",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)


def semantic_search(question, k=5):
    results = vectorstore.similarity_search(question, k=k)

    output = []

    for doc in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "group": doc.metadata.get("group")
        })

    return output