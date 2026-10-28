"""Deterministic safety checks and explicit production-readiness gaps."""
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src import chatbot
from src.safety import claim_amounts_supported, price_association_supported


URL = next(url for url in chatbot.get_approved_urls() if "/price-list/" in url)


def generated(claim, quote, source_id=1):
    return chatbot.GroundedAnswer(
        answerable=True,
        claims=[chatbot.EvidenceClaim(claim=claim, evidence_quote=quote, source_id=source_id)],
        medical_disclaimer_needed=False,
    )


def validate(claim, quote, content=None, source_id=1):
    return chatbot.validate_grounded_answer(
        generated(claim, quote, source_id),
        {1: {"content": content or quote, "source": URL, "title": "ITC prices"}},
        "What prices are listed for ITC?",
    )


@pytest.mark.parametrize("question", [
    "Can you book me an appointment tomorrow?",
    "What is the weather in Cancun?",
    "Is this treatment safe for my diabetes?",
])
def test_restricted_questions_never_invoke_retrieval_or_groq(question):
    with patch.object(chatbot, "prepare_evidence", side_effect=AssertionError), patch.object(
        chatbot, "get_llm", side_effect=AssertionError
    ):
        response = chatbot.ask_chatbot(question)
    assert response["status"] == "restricted"
    assert response["sources"] == []


def test_follow_up_passes_previous_user_question_to_retrieval_and_model():
    history = [
        {"role": "user", "content": "What does the adrenal cancer package include?"},
        {"role": "assistant", "content": "UNTRUSTED_ASSISTANT_FACT"},
    ]
    seen = {}

    def llm(prompt):
        seen["prompt"] = prompt.to_string()
        return chatbot.GroundedAnswer(answerable=False, claims=[], medical_disclaimer_needed=False)

    document = Document(page_content="There is no verified recovery duration here.", metadata={"source": URL})
    with patch.object(chatbot, "retrieve_documents", return_value=[(document, 0.1)]) as retrieve, patch.object(
        chatbot, "get_llm", return_value=RunnableLambda(llm)
    ):
        result = chatbot.ask_chatbot("And what about recovery time?", history)
    query = retrieve.call_args.args[0]
    assert "adrenal cancer" in query
    assert "recovery time" in query
    assert "adrenal cancer" in seen["prompt"]
    assert "UNTRUSTED_ASSISTANT_FACT" not in seen["prompt"]
    assert result["status"] == "unknown"


def test_retrieval_rejects_unapproved_urls():
    document = Document(page_content="A fabricated price list for the requested treatment.", metadata={"source": "https://example.com"})
    with patch.object(chatbot, "retrieve_documents", return_value=[(document, 0.1)]):
        assert chatbot.prepare_evidence("What is included?") == {}


def test_missing_evidence_returns_unknown_without_calling_groq():
    with patch.object(chatbot, "prepare_evidence", return_value={}), patch.object(
        chatbot, "get_llm", side_effect=AssertionError
    ):
        assert chatbot.ask_chatbot("What is included?")["status"] == "unknown"


def test_groq_failure_is_not_exposed_in_public_answer():
    with patch.object(chatbot, "prepare_evidence", return_value={1: {
        "content": "Approved package evidence.", "title": "Package", "source": URL,
    }}), patch.object(chatbot, "get_llm", side_effect=RuntimeError("PRIVATE_INTERNAL_DETAIL")):
        result = chatbot.ask_chatbot("What is included?")
    assert result["status"] == "error"
    assert "PRIVATE_INTERNAL_DETAIL" not in result["answer"]
    assert result["sources"] == []


def test_unknown_source_id_and_fabricated_quote_are_rejected():
    quote = "Breast Cancer | USD 18,995"
    assert validate("Breast Cancer: USD 18,995", quote, source_id=99)["status"] == "unknown"
    assert validate("Breast Cancer: USD 18,995", quote, content="No price is listed on this page.")["status"] == "unknown"


def test_explicit_currency_mismatch_is_rejected():
    assert not claim_amounts_supported("EUR 18,995", "USD 18,995", "Price in USD")


# Expected failures document reproducible limitations rather than claiming
# these safeguards exist. Keep strict=True so a future fix must update them.
@pytest.mark.xfail(strict=True, reason="Price qualifiers are prompted but not validated")
def test_starting_price_cannot_be_claimed_as_fixed_final_price():
    assert validate(
        "The final fixed cost is USD 18,995.",
        "Treatment starts from USD 18,995; price varies by duration.",
    )["status"] == "unknown"


@pytest.mark.xfail(strict=True, reason="Verbatim quotation is not semantic claim entailment")
def test_unrelated_claim_with_real_quote_is_rejected():
    assert validate(
        "The package guarantees a complete cure.",
        "The package includes a consultation with the clinic.",
    )["status"] == "unknown"


@pytest.mark.xfail(strict=True, reason="Unrecognized treatment names bypass the cancer-name allowlist")
def test_other_treatment_cannot_reuse_a_cancer_price():
    assert not price_association_supported(
        "Dental implants cost USD 18,995.", "Breast Cancer | USD 18,995",
    )


@pytest.mark.xfail(strict=True, reason="Dollar disambiguation does not handle mixed-currency contexts conservatively")
def test_mixed_currency_context_cannot_disambiguate_a_bare_dollar_price():
    assert not claim_amounts_supported(
        "USD 50", "$50", "The table contains both USD and MXN prices."
    )
