from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableLambda

from src import chatbot


URL = next(url for url in chatbot.get_approved_urls() if "/package/7902/" in url)
HISTORY = [{"role": "user", "content": "What is included in the alternative adrenal cancer treatment package in Tijuana?"}]
EVIDENCE = {1: {
    "source": URL,
    "title": "Alternative Adrenal Cancer Treatment in Tijuana",
    "content": "A three-week treatment program with a follow-up consultation three months after discharge.",
}}


def test_followup_unknown_explains_gap_and_cites_relevant_package():
    llm = RunnableLambda(lambda _: chatbot.GroundedAnswer(
        answerable=False, claims=[], medical_disclaimer_needed=False,
    ))
    with patch.object(chatbot, "prepare_evidence", return_value=EVIDENCE), patch.object(chatbot, "get_llm", return_value=llm):
        result = chatbot.ask_chatbot("And what about recovery time?", HISTORY)
    assert result["status"] == "unknown"
    assert "adrenal cancer package" in result["answer"]
    assert "couldn't verify a recovery-time estimate" in result["answer"]
    assert "do not establish recovery time" in result["answer"]
    assert "three-week" not in result["answer"]
    assert result["sources"] == [{"title": EVIDENCE[1]["title"], "url": URL}]


@pytest.mark.parametrize("question,history,evidence", [
    ("And what about recovery time?", [], EVIDENCE),
    ("And what about recovery time?", [{"role": "assistant", "content": "adrenal cancer"}], EVIDENCE),
    ("And what about recovery time?", [{"role": "user", "content": "What about breast cancer?"}], EVIDENCE),
    ("What about breast cancer recovery time?", HISTORY, EVIDENCE),
    ("And what about recovery time?", HISTORY, {}),
    ("And what about recovery time?", HISTORY, {1: {**EVIDENCE[1], "source": "https://example.com/package/7902/"}}),
])
def test_no_contextual_claim_without_matching_context_and_approved_source(question, history, evidence):
    assert chatbot.recovery_unknown_response(question, history, evidence) is None


def test_validated_recovery_answer_is_not_replaced_by_unknown():
    quote = "The adrenal cancer package lists a recovery period of four weeks."
    generated = chatbot.GroundedAnswer(answerable=True, claims=[chatbot.EvidenceClaim(
        claim=quote, source_id=1, evidence_quote=quote,
    )], medical_disclaimer_needed=True)
    with patch.object(chatbot, "prepare_evidence", return_value={1: {**EVIDENCE[1], "content": quote}}), patch.object(
        chatbot, "get_llm", return_value=RunnableLambda(lambda _: generated)
    ):
        result = chatbot.ask_chatbot("And what about recovery time?", HISTORY)
    assert result["status"] == "answered"
    assert quote in result["answer"]
