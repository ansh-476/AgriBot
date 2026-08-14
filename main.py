from retrieval_pipeline import semantic_search
from chatbot import generate_answer


def build_context(results):
    context = []

    for i, result in enumerate(results, 1):
        context.append(
            f"""
RESULT {i}
SOURCE: {result["source"]}
GROUP: {result["group"]}

{result["content"]}
"""
        )

    return "\n".join(context)


def main():
    print("=" * 70)
    print("AGRICULTURE RAG CHATBOT")
    print("=" * 70)
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching dataset...")

        results = semantic_search(question, k=5)

        if not results:
            print("\nAssistant: I could not find this information in the provided dataset.\n")
            continue

        context = build_context(results)

        print("Generating answer...")

        answer = generate_answer(
            question,
            context
        )

        print("\nAssistant:")
        print(answer)
        print()


if __name__ == "__main__":
    main()