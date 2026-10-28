
from langchain_core.documents import Document

from src.vector_store import (
    load_documents,
    split_documents
)


def test_seven_approved_documents():

    documents = load_documents()

    assert len(documents) == 7

    for document in documents:
        assert document.page_content.strip()
        assert document.metadata["source"].startswith(
            "https://"
        )


def test_chunking_preserves_sources():

    sample = Document(
        page_content=(
            "Example treatment information. " * 80
        ),
        metadata={
            "source": "https://example.com/test",
            "title": "Example Treatment"
        }
    )

    chunks = split_documents([sample])

    assert len(chunks) > 1

    for chunk in chunks:
        assert (
            chunk.metadata["source"]
            == "https://example.com/test"
        )
        assert chunk.page_content.strip()
