import json
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.chatbot import ask_chatbot, answer_listed_prices, build_search_query
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


HISTORY = [{"role": "user", "content": "What treatments are available at ITC?"}]


@pytest.mark.parametrize("question,history", [
    ("prices for breast cancer treatment", HISTORY),
    ("What is the price of breast cancer treatment?", HISTORY),
    ("price for breast cancer at ITC?", []),
    ("Prices for Breast Cancer Treatment at ITC Immunity Therapy Center", []),
])
def test_specific_price_selects_only_matching_row(question, history):
    result = answer_listed_prices(question, evidence(
        "Breast Cancer | Breast Cancer | $18995 | Chat\n"
        "Prostate Cancer | Prostate Cancer | $20000 | Chat"
    ), history)
    assert result["status"] == "answered"
    assert "Breast Cancer: USD 18,995" in result["answer"]
    assert "Prostate" not in result["answer"]
    assert result["sources"][0]["url"] == URL


@pytest.mark.parametrize("question,history", [
    ("prices for breast cancer treatment", []),
    ("prices for breast cancer treatment", [{"role": "assistant", "content": "ITC"}]),
    ("prices for breast cancer treatment", HISTORY + [{"role": "user", "content": "What about another clinic?"}]),
    ("prices for breast cancer treatment", [{"role": "user", "content": "Compare ITC versus another clinic"}]),
    ("prices for breast cancer treatment at another clinic", HISTORY),
    ("Does breast cancer treatment cost $50 at ITC?", HISTORY),
    ("prices for adrenal cancer treatment at ITC", []),
])
def test_specific_lookup_does_not_infer_unsupported_provider_or_treatment(question, history):
    assert answer_listed_prices(question, evidence(
        "Breast Cancer | Breast Cancer | $18995 | Chat"
    ), history) is None


def test_followup_uses_table_without_groq_and_preserves_qualifier():
    item = evidence("Breast Cancer | Breast Cancer | starting from $18995 | Chat")[1]
    document = Document(page_content=item["content"], metadata={"source": URL, "title": item["title"]})
    with patch("src.chatbot.retrieve_documents", return_value=[(document, 0.1)]) as retrieve, patch(
        "src.chatbot.get_llm", side_effect=AssertionError("Must answer from retrieved table")
    ):
        result = ask_chatbot("prices for breast cancer treatment", HISTORY)
    assert result["status"] == "answered"
    assert "starting from USD 18,995" in result["answer"]
    assert result["sources"][0]["url"] == URL
    query = retrieve.call_args.args[0]
    assert "ITC" in query and "breast cancer" in query and "price list" in query


def test_ordinary_followup_query_remains_contextual():
    query = build_search_query("And what about recovery time?", HISTORY)
    assert "Previous user question" in query and "recovery time" in query
