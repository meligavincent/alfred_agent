import datasets
from langchain.docstore.document import Document
import pandas as pd


df = pd.read_parquet("hf://datasets/agents-course/unit3-invitees/data/train-00000-of-00001.parquet")

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees",split="train")


# Login using e.g. `huggingface-cli login` to access this dataset


# Convert dataset entries into Document objects

docs = [
    Document(
        page_content="\n".join([
            f"Name:{guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description:{guest['description']}",
            f"Email: {guest['email']}"

        ]),
        metadata={"name":guest["name"]}
    )

    for guest in guest_dataset
]