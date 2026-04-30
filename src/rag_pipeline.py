from openai import OpenAI
from retriever import build_retriever, retrieve

client = OpenAI(api_key="key")


def expand_query(query):
    return [
        query,
        f"legal explanation of {query}",
        f"case law related to {query}",
        f"judgement reasoning for {query}"
    ]


def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a legal assistant.

Use ONLY the information from the retrieved documents.

If the information is insufficient, say:
"Not enough data available."

Retrieved Documents:
{context}

User Query:
{query}

Give a clear explanation and reasoning.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    index, texts = build_retriever()

    query = "Explain fraud-related legal cases based on available documents"

    queries = expand_query(query)

    all_docs = []
    for q in queries:
        docs = retrieve(q, index, texts)
        all_docs.extend(docs)

    unique_docs = list(set(all_docs))

    if len(unique_docs) == 0:
        print("No relevant documents found. Cannot generate reliable answer.")
    else:
        print("\n🔍 Retrieved Sources:\n")
        for i, doc in enumerate(unique_docs[:3]):
            print(f"Source {i+1}: {doc[:200]}...\n")

    
        answer = generate_answer(query, unique_docs)

        print("\n🧠 Final AI Answer:\n")
        print(answer)
