import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.chatbot import ask_chatbot, answer_listed_prices
from src.config import BASE_DIR
from src.safety import price_association_supported


QUESTION = "What prices are listed for ITC?"
URL = (
    "https://www.placidway.com/price-list/5/"
    "Alternative-Cancer-Treatment-by-ITC-Immunity-Therapy-Center"
)


def evidence(rows, url=URL):
    return {1: {
        "title": "ITC Immunity Therapy Center Prices",
        "source": url,
        "content": "Procedure | Details | Price in USD | Actions\n" + rows,
        "distance": 0.1,
    }}


@pytest.mark.parametrize("price", ["$18995", "$18,995", "USD 18,995"])
def test_listed_price_formats_and_citation(price):
    result = answer_listed_prices(QUESTION, evidence(
        f"Breast Cancer | Breast Cancer | {price} | Chat with Center"
    ))
    assert result["status"] == "answered"
    assert "Breast Cancer: USD 18,995" in result["answer"]
    assert result["sources"][0]["url"] == URL
    assert "Source evidence" not in result["answer"]


def test_rows_keep_their_own_prices_and_skip_unlabelled_rows():
    result = answer_listed_prices(QUESTION, evidence(
        "Breast Cancer | Breast Cancer | $18995 | Chat\n"
        "Prostate Cancer | Prostate Cancer | $20000 | Chat\n"
        " | | $50 | Chat"
    ))
    assert "Breast Cancer: USD 18,995" in result["answer"]
    assert "Prostate Cancer: USD 20,000" in result["answer"]
    assert "50" not in result["answer"]


def test_preserves_starting_price_qualifier():
    result = answer_listed_prices(QUESTION, evidence(
        "Breast Cancer | Breast Cancer | starting from $18995 | Chat"
    ))
    assert "starting from USD 18,995" in result["answer"]


@pytest.mark.parametrize("question", [
    "Does adrenal cancer cost $50 at ITC?",
    "Is ITC cheaper than Turkey?",
    "What prices are listed for another clinic?",
])
def test_does_not_answer_unsupported_questions_from_price_list(question):
    assert answer_listed_prices(question, evidence(
        "Breast Cancer | Breast Cancer | $18995 | Chat"
    )) is None


def test_rejects_unapproved_sources_and_conflicting_prices():
    rows = "Breast Cancer | Breast Cancer | $18995 | Chat"
    assert answer_listed_prices(QUESTION, evidence(rows, "https://example.com")) is None
    assert answer_listed_prices(QUESTION, evidence(
        rows + "\nBreast Cancer | Breast Cancer | $50 | Chat"
    )) is None


def test_repeated_questions_use_actual_ingested_table_without_llm():
    page = json.loads((BASE_DIR / "data/processed/clinic_price_list.json").read_text(encoding="utf-8"))
    document = Document(page_content=page["content"], metadata={
        "source": URL, "title": page["title"],
    })
    with patch("src.chatbot.retrieve_documents", return_value=[(document, 0.1)]), patch(
        "src.chatbot.get_llm", side_effect=AssertionError("Table lookup must not invoke Groq")
    ):
        history = []
        for _ in range(3):
            result = ask_chatbot(QUESTION, history)
            assert result["status"] == "answered"
            assert "USD 18,995" in result["answer"]
            assert result["sources"][0]["url"] == URL
            history.append({"role": "user", "content": QUESTION})


def test_all_named_treatments_must_match_their_price_row():
    assert not price_association_supported(
        "Breast Cancer and Prostate Cancer cost $18,995.",
        "Breast Cancer | $18995\nProstate Cancer | $50",
    )
