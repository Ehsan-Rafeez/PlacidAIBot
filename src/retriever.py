
"""
PlacidWay RAG Retriever

Searches the ChromaDB knowledge base and
returns relevant documents with similarity scores.
"""

from src.config import RETRIEVAL_TOP_K
from src.vector_store import load_vector_store


def retrieve_documents(
    query: str,
    top_k: int = RETRIEVAL_TOP_K
):
    """
    Retrieve relevant documents from ChromaDB.

    Parameters:
        query: User question or search query.
        top_k: Maximum number of results.

    Returns:
        List of (Document, distance) tuples.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query.strip(),
        k=top_k
    )

    return results


def display_results(results):
    """
    Print retrieved documents for testing.
    """

    if not results:
        print("No relevant documents found.")
        return

    for index, (document, score) in enumerate(
        results,
        start=1
    ):

        print("\n" + "=" * 60)

        print(f"Result #{index}")

        print(
            "Title:",
            document.metadata.get("title", "")
        )

        print(
            "Source:",
            document.metadata.get("source", "")
        )

        print(
            "Distance:",
            round(float(score), 4)
        )

        print("\nContent:\n")

        print(document.page_content)

        print("=" * 60)


if __name__ == "__main__":

    question = input(
        "Enter your question: "
    ).strip()

    results = retrieve_documents(question)

    display_results(results)
