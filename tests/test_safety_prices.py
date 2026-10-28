from decimal import Decimal

from src.chatbot import EvidenceClaim, GroundedAnswer, validate_grounded_answer
from src.safety import (
    claim_amounts_supported,
    monetary_values,
    price_association_supported,
)


SOURCE = (
    "ITC Immunity Therapy Center lists prices in USD. "
    "Alternative cancer treatment: USD 18,995."
)


def test_price_formats_match_when_source_establishes_usd():
    for claim in ("The listed price is $18995.", "The listed price is $18,995.", "The listed price is USD 18,995."):
        assert claim_amounts_supported(claim, claim, source_context=SOURCE)


def test_source_price_is_normalized_as_usd():
    assert monetary_values("USD 18,995") == {("USD", Decimal("18995"))}


def test_incorrect_price_is_rejected():
    assert not claim_amounts_supported(
        "The listed price is $50.",
        "Alternative cancer treatment: USD 18,995.",
        source_context=SOURCE,
    )


def test_price_cannot_be_attached_to_the_wrong_treatment():
    quote = "Breast Cancer | Breast Cancer | $18995 | Chat with Center"
    assert not price_association_supported(
        "Prostate Cancer costs $18,995.",
        quote,
    )


def test_price_treatment_association_requires_matching_row():
    quote = "Breast Cancer | Breast Cancer | $18995 | Chat with Center"
    assert price_association_supported(
        "Breast Cancer costs $18,995.",
        quote,
    )


def test_unsupported_answer_is_unknown():
    generated = GroundedAnswer(
        answerable=False,
        claims=[],
        medical_disclaimer_needed=False,
    )

    response = validate_grounded_answer(
        generated,
        {1: {"content": SOURCE, "source": "https://example.com", "title": "ITC", "distance": 0.1}},
        "What prices are listed for ITC?",
    )

    assert response["status"] == "unknown"


def test_citation_uses_the_validated_evidence_source():
    generated = GroundedAnswer(
        answerable=True,
        claims=[EvidenceClaim(
            claim="Breast Cancer costs $18,995.",
            source_id=1,
            evidence_quote=(
                "Breast Cancer | Breast Cancer | $18995 | Chat with Center"
            ),
        )],
        medical_disclaimer_needed=False,
    )
    source_url = "https://www.placidway.com/price-list/itc"
    response = validate_grounded_answer(
        generated,
        {1: {"content": SOURCE + "\nBreast Cancer | Breast Cancer | $18995 | Chat with Center", "source": source_url, "title": "ITC", "distance": 0.1}},
        "What prices are listed for ITC?",
    )

    assert response["status"] == "answered"
    assert response["sources"] == [{"title": "ITC", "url": source_url}]
    assert "Source evidence:" not in response["answer"]
