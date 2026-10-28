
import pytest

from src.ingestion import (
    normalize_url,
    extract_page_content
)


SAMPLE_HTML = """
<html>
<head>
    <title>Example Treatment Page</title>
</head>
<body>
    <header>Navigation links</header>

    <main>
        <h1>Example Treatment Package</h1>

        <p>
            This is a synthetic package
            used only for testing.
        </p>

        <h2>Package Includes</h2>

        <ul>
            <li>Consultation</li>
            <li>Follow-up discussion</li>
        </ul>

        <h2>Example Prices</h2>

        <table>
            <tr>
                <th>Treatment</th>
                <th>Price</th>
            </tr>
            <tr>
                <td>Example Package</td>
                <td>USD 1,000</td>
            </tr>
        </table>
    </main>

    <footer>Footer links</footer>
</body>
</html>
"""


def test_extract_page_content():

    result = extract_page_content(
        SAMPLE_HTML,
        "https://example.com/test"
    )

    assert result["title"] == "Example Treatment Page"

    assert "Consultation" in result["content"]

    assert "Follow-up discussion" in result["content"]

    assert "USD 1,000" in result["content"]

    assert "Example Package | USD 1,000" in (
        result["content"]
    )

    assert "Navigation links" not in result["content"]

    assert "Footer links" not in result["content"]


def test_normalize_url():

    result = normalize_url(
        "https://example.com/test#details"
    )

    assert result == "https://example.com/test"


def test_reject_http():

    with pytest.raises(ValueError):
        normalize_url("http://example.com/test")


def test_reject_invalid_url():

    with pytest.raises(ValueError):
        normalize_url("PASTE_EXACT_URL_1")
