
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def test_embeddings():

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    text = (
        "Alternative cancer treatment "
        "is available in Tijuana, Mexico."
    )

    vector = embeddings.embed_query(text)

    print("Original text:", text)

    print("Vector dimensions:", len(vector))

    print("First 5 vector values:", vector[:5])


if __name__ == "__main__":
    test_embeddings()
