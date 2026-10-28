
"""
PlacidWay website ingestion pipeline.

Features:
- Exact URL allowlist
- robots.txt compliance
- Identifiable User-Agent
- Polite request rate
- Redirect protection
- HTML and table extraction
- Source metadata
- Refresh and error reporting
"""

import hashlib
import json
import logging
import re
import time

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CRAWL_DELAY_SECONDS,
    USER_AGENT,
)


logger = logging.getLogger(__name__)

URL_CONFIG_PATH = BASE_DIR / "approved_urls.json"

BOT_NAME = "PlacidWayAssessmentBot"

TIMEOUT = (10, 30)

MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url):
    """
    Normalize a URL for exact allowlist comparison.
    Remove fragments but preserve paths and query strings.
    """
    parsed = urlsplit(url.strip())

    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            f"Only valid HTTPS URLs are allowed: {url}"
        )

    if parsed.port not in (None, 443):
        raise ValueError("Nonstandard ports are not allowed.")

    hostname = parsed.hostname.lower()

    path = parsed.path or "/"

    return urlunsplit((
        "https",
        hostname,
        path,
        parsed.query,
        ""
    ))


def load_approved_pages():
    """
    Read the seven explicitly approved URLs.
    """
    if not URL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing URL configuration: {URL_CONFIG_PATH}"
        )

    data = json.loads(
        URL_CONFIG_PATH.read_text(encoding="utf-8")
    )

    pages = data.get("pages", [])

    if len(pages) != 7:
        raise ValueError(
            "Exactly seven approved URLs are required."
        )

    seen_urls = set()
    seen_ids = set()

    for page in pages:
        page_id = page["id"]

        if not re.fullmatch(
            r"[a-zA-Z0-9_-]+", page_id
        ):
            raise ValueError(
                f"Invalid page ID: {page_id}"
            )

        url = normalize_url(page["url"])

        if url in seen_urls:
            raise ValueError(
                f"Duplicate approved URL: {url}"
            )

        if page_id in seen_ids:
            raise ValueError(
                f"Duplicate page ID: {page_id}"
            )

        seen_urls.add(url)
        seen_ids.add(page_id)

        page["url"] = url

    return pages


def clean_text(value):
    """
    Normalize whitespace without modifying facts.
    """
    return re.sub(r"\s+", " ", value).strip()


def extract_page_content(html, page_url):
    """
    Extract readable website content.

    Preserve headings, paragraphs, lists,
    and table rows.
    """
    soup = BeautifulSoup(html, "html.parser")

    title = (
        clean_text(soup.title.get_text(" ", strip=True))
        if soup.title
        else page_url
    )

    # Remove non-content elements.
    for element in soup.select(
        "script, style, noscript, template, "
        "nav, footer, header, form, "
        "svg, iframe"
    ):
        element.decompose()

    # Prefer the main content area.
    root = (
        soup.select_one("main")
        or soup.select_one("article")
        or soup.select_one('[role="main"]')
        or soup.body
        or soup
    )

    blocks = []

    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6",
         "p", "li", "table"]
    ):

        # Tables are handled as a complete unit.
        # Skip elements nested inside tables.
        if (
            element.name != "table"
            and element.find_parent("table")
        ):
            continue

        if element.name == "table":
            rows = []

            for tr in element.find_all("tr"):
                cells = [
                    clean_text(
                        cell.get_text(" ", strip=True)
                    )
                    for cell in tr.find_all(
                        ["th", "td"], recursive=False
                    )
                ]

                if cells:
                    rows.append(" | ".join(cells))

            if rows:
                blocks.append(
                    "TABLE:\n" + "\n".join(rows)
                )

            continue

        # Avoid duplicate text from nested list items.
        if element.name == "li":
            for nested_list in element.find_all(
                ["ul", "ol"]
            ):
                nested_list.decompose()

        text = clean_text(
            element.get_text(" ", strip=True)
        )

        if not text:
            continue

        if element.name.startswith("h"):
            blocks.append(
                f"\n## {text}"
            )
        elif element.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    # Fallback if structured extraction is empty.
    if not blocks:
        fallback = clean_text(
            root.get_text(" ", strip=True)
        )
        if fallback:
            blocks = [fallback]

    # Remove consecutive identical blocks.
    deduplicated = []

    for block in blocks:
        if not deduplicated or block != deduplicated[-1]:
            deduplicated.append(block)

    content = "\n".join(deduplicated).strip()

    return {
        "title": title,
        "content": content
    }


class WebsiteCrawler:
    """
    Controlled crawler for explicitly approved URLs.
    """

    def __init__(self, approved_pages):

        self.approved_pages = approved_pages

        self.allowed_urls = {
            normalize_url(page["url"])
            for page in approved_pages
        }

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml"
        })

        self.robots_cache = {}

        # Track rate limiting by hostname.
        self.last_request = {}

    def polite_wait(self, hostname, delay):
        """
        Ensure requests are appropriately spaced.
        """
        last = self.last_request.get(hostname)

        if last is not None:
            elapsed = time.monotonic() - last
            remaining = delay - elapsed

            if remaining > 0:
                time.sleep(remaining)

        self.last_request[hostname] = time.monotonic()

    def fetch_robots(self, url):
        """
        Fetch and parse robots.txt once per origin.

        Fail closed if robots.txt cannot be verified.
        """
        parsed = urlsplit(url)

        origin = (
            f"{parsed.scheme}://{parsed.netloc}"
        )

        if origin in self.robots_cache:
            return self.robots_cache[origin]

        robots_url = origin + "/robots.txt"

        self.polite_wait(
            parsed.hostname,
            CRAWL_DELAY_SECONDS
        )

        response = self.session.get(
            robots_url,
            timeout=TIMEOUT,
            allow_redirects=False
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Cannot verify robots.txt: "
                f"{robots_url} "
                f"(HTTP {response.status_code})"
            )

        parser = RobotFileParser()
        parser.parse(response.text.splitlines())

        self.robots_cache[origin] = parser

        return parser

    def get_delay(self, parser):
        """
        Honor crawl delay and request rate rules.
        """
        crawl_delay = parser.crawl_delay(BOT_NAME)

        request_rate = parser.request_rate(BOT_NAME)

        delay = CRAWL_DELAY_SECONDS

        if crawl_delay is not None:
            delay = max(delay, crawl_delay)

        if request_rate is not None:
            if request_rate.requests > 0:
                rate_delay = (
                    request_rate.seconds
                    / request_rate.requests
                )
                delay = max(delay, rate_delay)

        return delay

    def fetch_page(self, url):
        """
        Download a permitted page.

        Every redirect target must also be approved.
        """
        current_url = normalize_url(url)

        for _ in range(MAX_REDIRECTS + 1):

            if current_url not in self.allowed_urls:
                raise PermissionError(
                    f"URL is not approved: {current_url}"
                )

            parser = self.fetch_robots(current_url)

            if not parser.can_fetch(
                BOT_NAME,
                current_url
            ):
                raise PermissionError(
                    f"Blocked by robots.txt: {current_url}"
                )

            parsed = urlsplit(current_url)

            self.polite_wait(
                parsed.hostname,
                self.get_delay(parser)
            )

            response = self.session.get(
                current_url,
                timeout=TIMEOUT,
                allow_redirects=False,
                stream=True
            )

            try:
                # Handle redirects manually.
                if response.status_code in (
                    301, 302, 303, 307, 308
                ):
                    location = response.headers.get(
                        "Location"
                    )

                    if not location:
                        raise RuntimeError(
                            "Redirect has no Location header."
                        )

                    next_url = normalize_url(
                        urljoin(current_url, location)
                    )

                    if next_url not in self.allowed_urls:
                        raise PermissionError(
                            "Redirect target is not approved: "
                            + next_url
                        )

                    current_url = next_url
                    continue

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" not in content_type:
                    raise ValueError(
                        "Expected HTML, received: "
                        + content_type
                    )

                # Limit the maximum downloaded size.
                chunks = []
                total_bytes = 0

                for chunk in response.iter_content(
                    chunk_size=65536
                ):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_RESPONSE_BYTES:
                        raise ValueError(
                            "HTML response exceeds size limit."
                        )

                    chunks.append(chunk)

                raw_html = b"".join(chunks)

                response._content = raw_html

                if not response.encoding:
                    response.encoding = (
                        response.apparent_encoding
                        or "utf-8"
                    )

                return response.text, current_url

            finally:
                response.close()

        raise RuntimeError(
            "Maximum permitted redirects exceeded."
        )


def save_document(page, html, extracted, source_url):
    """
    Save HTML and cleaned JSON document.
    """
    RAW_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    page_id = page["id"]

    content = extracted["content"]

    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    document = {
        "id": page_id,
        "page_type": page["type"],
        "title": extracted["title"],
        "source_url": source_url,
        "fetched_at": utc_now(),
        "content_hash": content_hash,
        "content": content,
        "character_count": len(content)
    }

    html_path = RAW_DATA_DIR / f"{page_id}.html"

    json_path = (
        PROCESSED_DATA_DIR / f"{page_id}.json"
    )

    html_path.write_text(
        html, encoding="utf-8"
    )

    json_path.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    return document


def run_ingestion():
    """
    Run ingestion for the seven approved pages.

    Failed pages are reported, not fabricated.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    pages = load_approved_pages()

    crawler = WebsiteCrawler(pages)

    report = {
        "started_at": utc_now(),
        "total_pages": len(pages),
        "successful": 0,
        "failed": 0,
        "results": []
    }

    for page in pages:

        page_id = page["id"]
        url = page["url"]

        logger.info(
            "Processing %s: %s",
            page_id, url
        )

        try:
            html, final_url = crawler.fetch_page(url)

            extracted = extract_page_content(
                html,
                final_url
            )

            if len(extracted["content"]) < 100:
                raise ValueError(
                    "Extracted content is too short. "
                    "Manual review required."
                )

            document = save_document(
                page,
                html,
                extracted,
                final_url
            )

            report["successful"] += 1

            report["results"].append({
                "id": page_id,
                "status": "success",
                "source_url": final_url,
                "characters": document[
                    "character_count"
                ],
                "content_hash": document[
                    "content_hash"
                ]
            })

            logger.info(
                "Saved %s (%s characters)",
                page_id,
                document["character_count"]
            )

        except (
            requests.RequestException,
            ValueError,
            PermissionError,
            RuntimeError
        ) as error:

            report["failed"] += 1

            report["results"].append({
                "id": page_id,
                "status": "failed",
                "source_url": url,
                "error": str(error)
            })

            logger.error(
                "Failed %s: %s",
                page_id,
                error
            )

    report["finished_at"] = utc_now()

    PROCESSED_DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    report_path = (
        PROCESSED_DATA_DIR / "ingestion_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    logger.info(
        "Ingestion finished: %s successful, %s failed",
        report["successful"],
        report["failed"]
    )

    return report


if __name__ == "__main__":
    result = run_ingestion()

    if result["failed"]:
        raise SystemExit(1)
