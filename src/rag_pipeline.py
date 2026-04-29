from openai import OpenAI
from retriever import build_retriever, retrieve


client = OpenAI(api_key="key")

def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
You are a legal assistant.

Based on the following retrieved documents:
{context}

Answer the query:
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

    query = "fraud case legal judgement"

    docs = retrieve(query, index, texts)

    answer = generate_answer(query, docs)

    print("\nFinal AI Answer:\n")
    print(answer)
