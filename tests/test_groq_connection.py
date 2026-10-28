
from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    validate_config
)


def test_groq():
    # Validate configuration
    validate_config()

    print("Connecting to Groq API...")

    # Initialize the model
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )

    # Send a test prompt
    response = llm.invoke(
        "Reply with exactly: Groq API connected successfully"
    )

    print("Groq Response:")
    print(response.content)


if __name__ == "__main__":
    test_groq()
