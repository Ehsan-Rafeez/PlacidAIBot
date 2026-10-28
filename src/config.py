
import os
from pathlib import Path

from dotenv import load_dotenv

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 4

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

CHROMA_DB_DIR = BASE_DIR / "chroma_db"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# Embedding model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Vector database
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "chroma_db"
)

CHROMA_DB_DIR = BASE_DIR / CHROMA_DB_PATH


# Data directories
DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Chunking configuration
CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150


# Retriever configuration
RETRIEVAL_TOP_K = 4


# Website crawling configuration
CRAWL_DELAY_SECONDS = 1.0

USER_AGENT = (
    "PlacidWayAssessmentBot/1.0 "
    "(Educational website knowledge crawler)"
)


def validate_config():
    """
    Verify required API configuration.
    """

    if (
        not GROQ_API_KEY
        or GROQ_API_KEY == "your_groq_api_key_here"
    ):
        raise ValueError(
            "Missing GROQ_API_KEY. "
            "Add it to the .env file."
        )
