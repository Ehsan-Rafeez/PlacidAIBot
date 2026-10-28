
"""
PlacidWay vector knowledge base.

Loads the seven verified JSON documents,
splits them into chunks, embeds them, and
builds a persistent ChromaDB index.
"""

import json
import os
import re
import uuid

from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

COLLECTION_NAME = "placidway_knowledge"
MANIFEST_PATH = CHROMA_DB_DIR / "active_index.json"


def load_documents():
    """
    Load exactly the seven approved,
    successfully extracted documents.
    """

    approved_file = BASE_DIR / "approved_urls.json"

    approved_config = json.loads(
        approved_file.read_text(encoding="utf-8")
    )

    approved_pages = approved_config["pages"]

    if len(approved_pages) != 7:
        raise ValueError(
            "Expected exactly seven approved pages."
        )

    report_path = (
        PROCESSED_DATA_DIR / "ingestion_report.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(
            "Ingestion report missing. Run Phase 3."
        )

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    if (
        report.get("successful") != 7
        or report.get("failed") != 0
    ):
        raise ValueError(
            "Ingestion must have 7 successes "
            "and 0 failures before indexing."
        )

    successful_ids = {
        item["id"]
        for item in report.get("results", [])
        if item.get("status") == "success"
    }

    expected_ids = {
        page["id"]
        for page in approved_pages
    }

    if successful_ids != expected_ids:
        raise ValueError(
            "Ingestion report does not match "
            "approved page IDs."
        )

    documents = []

    for page in approved_pages:

        path = (
            PROCESSED_DATA_DIR
            / f"{page['id']}.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing document: {path}"
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            data.get("id") != page["id"]
            or data.get("source_url") != page["url"]
            or data.get("page_type") != page["type"]
        ):
            raise ValueError(
                f"Metadata mismatch: {path.name}"
            )

        content = data.get("content", "").strip()

        if len(content) < 100:
            raise ValueError(
                f"Insufficient content: {page['id']}"
            )

        # Do not allow empty or unverified source URLs.
        if not data.get("source_url", "").startswith(
            "https://"
        ):
            raise ValueError(
                f"Invalid source URL: {page['id']}"
            )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "page_id": page["id"],
                    "page_type": page["type"],
                    "source": data["source_url"],
                    "title": data.get("title", ""),
                    "content_hash": data.get(
                        "content_hash", ""
                    )
                }
            )
        )

    return documents


def split_documents(documents):
    """
    Split documents into searchable chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

        # Keep useful source context with the text.
        title = chunk.metadata.get("title", "")

        if title and not chunk.page_content.startswith(
            title
        ):
            chunk.page_content = (
                f"Page: {title}\n"
                + chunk.page_content
            )

    return chunks


def get_embeddings():
    """
    Load the local Hugging Face embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def build_vector_store():
    """
    Build a new index without modifying
    the previously active index.

    Activate it only after successful indexing.
    """

    print("Loading approved documents...")

    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks produced from documents."
        )

    print(f"Generated chunks: {len(chunks)}")

    print("Loading local embedding model...")

    embeddings = get_embeddings()

    CHROMA_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    index_id = uuid.uuid4().hex

    relative_path = f"indexes/{index_id}"

    index_path = CHROMA_DB_DIR / relative_path

    print("Creating ChromaDB index...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(index_path),
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    chunk_ids = [
        f"{index_id}_{i}"
        for i in range(len(chunks))
    ]

    # Insert chunks in small batches.
    batch_size = 32

    for start in range(0, len(chunks), batch_size):

        end = start + batch_size

        vector_store.add_documents(
            documents=chunks[start:end],
            ids=chunk_ids[start:end]
        )

    stored_count = vector_store._collection.count()

    if stored_count != len(chunks):
        raise RuntimeError(
            "ChromaDB document count mismatch."
        )

    print(f"Stored vectors: {stored_count}")

    manifest = {
        "index_id": index_id,
        "index_path": relative_path,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "source_count": len(documents),
        "chunk_count": len(chunks),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    # Atomically update the active index pointer.
    temporary_manifest = (
        CHROMA_DB_DIR / f"manifest_{index_id}.tmp"
    )

    temporary_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    os.replace(
        temporary_manifest,
        MANIFEST_PATH
    )

    print("Knowledge base activated successfully!")

    return manifest


def load_vector_store():
    """
    Load the currently active persistent index.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Knowledge base not built yet. "
            "Run python -m src.vector_store"
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if manifest["embedding_model"] != EMBEDDING_MODEL:
        raise ValueError(
            "Embedding model changed. "
            "Rebuild the knowledge base."
        )

    relative_path = Path(manifest["index_path"])

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError(
            "Invalid index path in manifest."
        )

    index_path = CHROMA_DB_DIR / relative_path

    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing Chroma index: {index_path}"
        )

    embeddings = get_embeddings()

    return Chroma(
        collection_name=manifest["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(index_path),
        create_collection_if_not_exists=False
    )


if __name__ == "__main__":
    build_vector_store()
