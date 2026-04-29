from datasets import load_dataset
import pandas as pd
import os

def download_dataset():
    print("Downloading dataset...")

    dataset = load_dataset("ag_news")  

    data = dataset["train"]

    texts = []
    labels = []

    for item in data:
        texts.append(item["text"])
        labels.append(str(item["label"]))

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/legal_cases.csv", index=False)

    print("Dataset saved successfully!")

if __name__ == "__main__":
    download_dataset()
