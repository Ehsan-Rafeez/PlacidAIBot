# PlacidWay AI Assistant

A Python chatbot that answers questions using the seven PlacidWay pages listed in [`approved_urls.json`](approved_urls.json). It uses LangChain, Groq, Hugging Face embeddings, ChromaDB, and a Streamlit chat interface.

Supported answers include approved source links. The assistant declines appointment booking and personalized medical advice, and returns an honest fallback when it cannot verify an answer. It is an assessment prototype; see [known limitations](#known-limitations) before public deployment.

## Prerequisites

- Python **3.11** (the version used for this project's assessment).
- A Groq API key and access to the configured model in your Groq account.
- Internet access to install dependencies, fetch the approved pages, download embedding weights, and call Groq.
- A writable local directory for scraped data and ChromaDB. A GPU is not required; embeddings run on CPU.

## 1. Create a virtual environment

Download or clone this project, then open a terminal in the directory containing `app.py` and `requirements.txt`.

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python directly for every command below, for example:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the following steps from the project root with that environment activated.

## 2. Configure API access

Copy the example configuration once, if you do not already have `.env`:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Open `.env` in your editor and replace the placeholder with your own **Groq** key. You can create a key in the [Groq Console](https://console.groq.com/keys).

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_PATH=chroma_db

# Optional: leave empty if you do not have a Hugging Face token.
HF_TOKEN=
```

Keep each value on a single line. Keep the model and path settings populated: an empty value overrides the Python defaults. Model access can vary by account; change `GROQ_MODEL` to a model available to your account that supports the app's structured output if necessary.

`.env` is excluded by `.gitignore`. Never commit or share real API keys. The Groq key stays on the Python server; prompts and retrieved passages are sent to Groq for generated answers. Restart the app after changing environment settings.

## 3. Prepare the seven-page knowledge base

Raw pages, processed documents, and the vector database are local generated files excluded from version control. A fresh installation must create them:

```powershell
python -m src.ingestion
```

This fetches only the configured approved pages and writes `data/raw/`, `data/processed/`, and `data/processed/ingestion_report.json`. Check that the report shows **7 successful, 0 failed** before continuing. If a page is blocked, unavailable, or its content has changed, resolve the ingestion failure instead of proceeding with incomplete evidence.

Build the searchable index:

```powershell
python -m src.vector_store
```

The first build downloads the embedding model and can take several minutes. The indexer should report seven loaded documents and successful activation. It writes a persistent index and `chroma_db/active_index.json` (or the directory configured by `CHROMA_DB_PATH`).

You do **not** need to ingest and rebuild on every app launch. Repeat these two steps when refreshing source content. Rebuild after changing the embedding model or chunking configuration. Keep the database directory and active manifest together; on hosted environments, place them on durable storage.

## 4. Start the chatbot

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally **http://localhost:8501**. Stop the server with `Ctrl+C`.

The top-right corner shows **v1.1** for this release. If a deployment shows an older version or no badge, verify that the updated code was deployed and the server restarted. Future releases can update `APP_VERSION` in `src/version.py`.

Try:

- What prices are listed for ITC?
- What is included in the alternative adrenal cancer treatment package?
- And what about recovery time?
- Can you book me an appointment tomorrow?

The sidebar's **Clear Conversation** button starts a new conversation. Existing messages retain their original answers after code changes.

## Tests and diagnostics

Run the automated suite after configuration and ingestion:

```powershell
python -m pytest -q
```

The full suite includes live Groq and Hugging Face checks, so it requires network/API access and can consume API quota. For deterministic grounding and safety checks without model downloads or live Groq calls:

```powershell
python -m pytest tests/test_safety_prices.py tests/test_listed_prices.py tests/test_assessment_safety.py -q
```

These tests still require the dependencies and ingested local documents. Some safeguards are marked `xfail` because they expose known limitations; `xfail` is not a passing safeguard.

Useful checks:

```powershell
# Verify Groq connectivity (uses API quota).
python -m tests.test_groq_connection

# Inspect extracted pricing and Chroma retrieval; no Groq calls.
python -m tests.diagnose_price

# Run the eight live assessment scenarios and save actual responses.
python -m tests.evaluate_chatbot
```

The assessment overwrites `test_results/assessment_results.json` with the current run. Newly generated results need renewed source review; earlier review notes do not approve new answers.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named 'src'` | Run from the project root using module syntax, e.g. `python -m tests.diagnose_price`, rather than running the test file path directly. |
| `Relative module names not supported` | `-m` takes dotted module names, without `./`, backslashes, or `.py`. |
| `Missing GROQ_API_KEY` or authentication failure | Set your Groq key in `.env`; verify the model is accessible to your account. An OpenAI key cannot authenticate with Groq. |
| `Illegal header value` | Check for accidental newlines or whitespace in the API key. Do not print the key to debug it. |
| Hugging Face unauthenticated-request warning | Usually informational. Set the optional `HF_TOKEN` for authenticated downloads. `Loading weights: 100%` indicates the weights loaded; it does not prove the full request succeeded. |
| Knowledge base missing | Run ingestion, confirm seven successes, then run `python -m src.vector_store`. |
| Embedding model changed | Rebuild the index with the configured embedding model. |
| Answer is `unknown` | Review retrieved evidence with the pricing diagnostic. The source may lack an answer, or validation may have rejected a model quote, amount, currency, or association. Do not disable validation to force an answer. |
| Old answers remain after a fix | Restart Streamlit and clear the conversation, then submit a new question. |

## Project layout

```text
app.py                        Streamlit UI and session history
approved_urls.json            Seven-page source allowlist
src/config.py                 Environment configuration
src/ingestion.py              Page extraction and ingestion report
src/vector_store.py           Chunking, embeddings, persistent Chroma index
src/retriever.py               Semantic retrieval
src/chatbot.py                 Grounded generation and ITC price-table lookup
src/safety.py                  Safety routing and quote/price validation
tests/                        Automated tests and assessment scripts
test_results/                 Assessment reports
```

## Known limitations

The current prototype has known gaps in semantic claim validation, preservation of price qualifiers, general treatment-price association, and mixed-currency disambiguation. Exact quotation matching alone does not guarantee that a claim follows from the quote. Embeddings and vector-store resources also reload per retrieval, affecting latency.

The [production-readiness review](test_results/production_readiness.md) lists recommended fixes for grounding, caching, persistence, Groq failures, API security, and deployment. The [saved assessment](test_results/assessment_results.json) records a particular local source snapshot; it is not a guarantee of current website content or medical accuracy. Review these before exposing the app publicly.
