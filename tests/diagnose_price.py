
"""
Diagnose PlacidWay pricing extraction and retrieval.

No Groq API calls.
"""

import json
import re

from src.config import PROCESSED_DATA_DIR
from src.retriever import retrieve_documents
from src.safety import monetary_values


PRICE_PATTERN = re.compile(
    r"\$\s*[\d,]+(?:\.\d{1,2})?"
    r"|USD\s*[\d,]+(?:\.\d{1,2})?"
    r"|\b[\d,]+\s*USD\b",
    re.IGNORECASE
)


def check_extracted_prices():

    print("\n=== STEP 1: EXTRACTED PRICES ===")

    for page_id in [
        "clinic_price_list",
        "clinic_profile"
    ]:

        path = (
            PROCESSED_DATA_DIR
            / f"{page_id}.json"
        )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        content = data["content"]

        print(f"\nPage: {page_id}")
        print("Characters:", len(content))

        matches = list(
            PRICE_PATTERN.finditer(content)
        )

        print("Price matches:", len(matches))

        for match in matches[:12]:

            start = max(0, match.start() - 100)
            end = min(
                len(content),
                match.end() + 100
            )

            snippet = content[start:end]

            print("\n", snippet.replace("\n", " "))


def check_retrieval():

    print("\n=== STEP 2: RETRIEVAL ===")

    questions = [
        "What prices are listed for ITC?",
        "ITC Immunity Therapy Center "
        "cancer treatment price list USD",
        "How much does cancer treatment "
        "at ITC cost in Tijuana?"
    ]

    for question in questions:

        print("\nQUESTION:", question)

        results = retrieve_documents(
            question,
            top_k=8
        )

        for document, distance in results:

            page_id = document.metadata.get(
                "page_id", ""
            )

            amounts = monetary_values(
                document.page_content
            )

            print(
                f"Page: {page_id} | "
                f"Distance: {distance:.4f} | "
                f"Amounts: {sorted(amounts)}"
            )


if __name__ == "__main__":

    check_extracted_prices()
    check_retrieval()
