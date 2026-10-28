
"""
Safety and evidence validation for PlacidWay RAG.
"""

import re
from decimal import Decimal
from urllib.parse import urlsplit


UNKNOWN_RESPONSE = (
    "I couldn't find enough information in the approved "
    "PlacidWay pages to answer that reliably. "
    "Please contact PlacidWay to request a free quote."
)

MEDICAL_RESPONSE = (
    "I can't determine whether a treatment is safe or "
    "appropriate for your individual medical condition. "
    "Please discuss your medical history, risks, and "
    "treatment options with a qualified doctor."
)

BOOKING_RESPONSE = (
    "I cannot book appointments directly. "
    "Please contact PlacidWay through its official "
    "website to request an appointment or free quote."
)

OUT_OF_SCOPE_RESPONSE = (
    "I can help with treatments, clinics, packages, "
    "and costs described in the approved PlacidWay "
    "pages. I don't have verified information "
    "to answer that question."
)


# Conservative patterns for routing clearly
# unsupported actions and safety-sensitive questions.
BOOKING_PATTERN = re.compile(
    r"\b(book|schedule|reserve|confirm|cancel|reschedule)"
    r"\b.{0,45}\b(appointment|consultation|visit)\b",
    re.IGNORECASE
)

MEDICAL_PATTERN = re.compile(
    r"\b("
    r"safe for me|safe for my|"
    r"should i undergo|should i get|"
    r"can i undergo|can i stop taking|"
    r"should i stop taking|"
    r"recommend.*dosage|"
    r"my diabetes|my pregnancy|"
    r"my medication|my medical condition|"
    r"diagnose me"
    r")\b",
    re.IGNORECASE
)

OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(weather|temperature|football score|"
    r"stock market|bitcoin price)\b",
    re.IGNORECASE
)


PRICE_INTENT_PATTERN = re.compile(
    r"\b("
    r"price|prices|pricing|"
    r"cost|costs|cheaper|cheap|"
    r"expensive|affordable|"
    r"quote|fee|fees|"
    r"how much|"
    r"compare prices|"
    r"usd|eur|mxn"
    r")\b|[$€£]",
    re.IGNORECASE
)


# Numeric monetary values with an explicit currency.
CURRENCY_AMOUNT_PATTERN = re.compile(
    r"(?<!\w)(?:USD|MXN|EUR|GBP|CAD|AUD|PKR|"
    r"US\$|MX\$|\$|€|£)\s*"
    r"\d[\d,]*(?:\.\d{1,2})?"
    r"|\b\d[\d,]*(?:\.\d{1,2})?\s*"
    r"(?:USD|MXN|EUR|GBP|CAD|AUD|PKR)\b",
    re.IGNORECASE
)


def classify_question(question):
    """
    Return a direct safety response when needed.

    None means the question can proceed to retrieval.
    """
    text = question.strip()

    if BOOKING_PATTERN.search(text):
        return BOOKING_RESPONSE

    if MEDICAL_PATTERN.search(text):
        return MEDICAL_RESPONSE

    if OUT_OF_SCOPE_PATTERN.search(text):
        return OUT_OF_SCOPE_RESPONSE

    return None


def is_price_question(question):
    return bool(PRICE_INTENT_PATTERN.search(question))


def normalize_text(text):
    """
    Normalize whitespace for exact quote matching.
    """
    return " ".join(text.split())


def quote_is_supported(quote, original_content):
    """
    Ensure cited evidence is an actual substring
    of the retrieved page passage.
    """
    quote = normalize_text(quote)
    content = normalize_text(original_content)

    return bool(
        len(quote) >= 12
        and quote.casefold() in content.casefold()
    )



def normalize_money(value, context=""):
    """
    Normalize monetary values for comparison.

    Examples:
        USD 18,995 -> ("USD", Decimal("18995"))
        18995 USD  -> ("USD", Decimal("18995"))
        $18,995    -> ("USD", Decimal("18995"))
                       only if context establishes USD

    An ambiguous $ symbol remains unspecified
    if the source doesn't identify its currency.
    """

    text = value.upper().strip()
    text = text.replace(",", "")

    context_upper = context.upper()

    currency = None

    if "USD" in text or "US$" in text:
        currency = "USD"

    elif "MXN" in text or "MX$" in text:
        currency = "MXN"

    elif "EUR" in text or "€" in text:
        currency = "EUR"

    elif "GBP" in text or "£" in text:
        currency = "GBP"

    elif "$" in text:
        # A dollar sign is safe to treat as USD only when the
        # surrounding source explicitly establishes USD.
        if re.search(r"\bUSD\b", context_upper):
            currency = "USD"
        elif "MXN" in context_upper:
            currency = "MXN"
        else:
            currency = "UNSPECIFIED_$"

    else:
        for code in ("CAD", "AUD", "PKR"):
            if code in text:
                currency = code
                break

    number_match = re.search(
        r"\d+(?:\.\d{1,2})?",
        text
    )

    if not number_match:
        return None

    amount = Decimal(number_match.group())

    return currency, amount


def claim_amounts_supported(
    claim,
    evidence,
    source_context=""
):
    """
    Validate that every monetary amount in a claim
    exists in the supporting evidence.

    monetary_values() returns:
        {("USD", Decimal("18995"))}

    Currency must match. No conversions or
    inferred prices are allowed.
    """

    # Apply the source context to both sides. This is important when
    # the model writes "$18,995" while the source says "USD 18,995".
    claim_values = monetary_values(claim, context=source_context)
    evidence_values = monetary_values(evidence, context=source_context)

    # Context can establish that a dollar symbol
    # in the source represents US dollars.
    context_upper = source_context.upper()

    if (
        "PRICE IN USD" in context_upper
        or "PRICES IN USD" in context_upper
        or "USD" in context_upper
    ):
        evidence_values = {
            (
                "USD" if currency in (
                    "$", "UNSPECIFIED_$"
                ) else currency,
                amount
            )
            for currency, amount in evidence_values
        }

    # The claim must not introduce any new amount.
    return claim_values.issubset(evidence_values)


def price_association_supported(claim, evidence_quote):
    """
    Keep treatment names tied to the price row that supports them.

    Claims that contain only a price do not assert a treatment
    association and remain valid. When a claim names a treatment,
    require a meaningful two-word phrase from that claim to appear
    in the quoted source row.
    """
    if not monetary_values(claim):
        return True

    claim_text = CURRENCY_AMOUNT_PATTERN.sub(" ", claim)
    known_treatment_names = {
        "breast", "prostate", "lung", "colon", "colorectal",
        "pancreatic", "ovarian", "cervical", "liver", "kidney",
        "leukemia", "lymphoma", "melanoma", "brain", "stomach",
        "esophageal", "sarcoma", "bone",
    }
    treatment_phrases = {
        f"{prefix.casefold()} cancer"
        for prefix in re.findall(
            r"\b([A-Za-z][A-Za-z'-]*)\s+cancer\b",
            claim_text,
            re.IGNORECASE,
        )
        if prefix.casefold() in known_treatment_names
    }

    # Generic claims such as "alternative cancer treatment" do not
    # name a particular treatment and therefore need no row match.
    if not treatment_phrases:
        return True

    # Every named treatment must occur in a price-bearing row with the
    # claimed amount, rather than anywhere in a multi-row quotation.
    claim_amounts = {amount for _, amount in monetary_values(claim)}
    return all(
        any(
            phrase in normalize_text(row).casefold()
            and claim_amounts.issubset({
                amount for _, amount in monetary_values(row)
            })
            for row in evidence_quote.splitlines()
        )
        for phrase in treatment_phrases
    )

def monetary_values(text, context=""):
    """
    Extract currency-denominated monetary values.
    """
    return {
        normalize_money(match.group(0), context=context)
        for match in CURRENCY_AMOUNT_PATTERN.finditer(text)
    }




def validate_source_url(url, approved_urls):
    """
    Only allow URLs from the approved source list.
    """
    return (
        urlsplit(url).scheme == "https"
        and url in set(approved_urls)
    )


def make_response(answer, sources=None, status="answered"):
    """
    Standard response format used by chatbot and UI.
    """
    return {
        "answer": answer,
        "sources": sources or [],
        "status": status
    }
