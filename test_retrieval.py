from retrieval_pipeline import semantic_search

question = input("Enter your agriculture question: ")

results = semantic_search(question, k=5)

print("\n" + "=" * 70)
print("RETRIEVAL RESULTS")
print("=" * 70)

for i, result in enumerate(results, 1):
    print(f"\nRESULT {i}")
    print("SOURCE:", result["source"])
    print("GROUP:", result["group"])
    print("-" * 70)
    print(result["content"][:2000])