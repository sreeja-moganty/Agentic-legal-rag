from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

def build_retriever():
    print("Loading dataset...")
    df = pd.read_csv("data/legal_cases.csv")
    df = df.sample(2000)
    texts = df["text"].fillna("").tolist()

    print("Creating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts

def retrieve(query, index, texts, top_k=3):
    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = [texts[i] for i in indices[0]]

    return results


if __name__ == "__main__":
    index, texts = build_retriever()

    query = "property dispute fraud case"

    results = retrieve(query, index, texts)

    print("\nTop Results:\n")
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:\n", res[:500])
