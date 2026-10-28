# PlacidWay final assessment — 27 October 2026 (Asia/Karachi)

## Result

All eight actual assessment scenarios passed the stated behavior and source checks in this run. The original automated suite passed 26 tests. The expanded suite passed 35 tests, with zero unexpected failures and four strict expected failures exposing missing production safeguards. Those four are **known failing safeguards**, not passes. Runtime chatbot functionality was not rewritten for this assessment.

The JSON report contains actual responses, elapsed times, input history, retrieved passages, structured model claims, validation rejection messages, per-check results, and hashes identifying all seven local approved-page snapshots. No other pages supplied factual evidence. Links were checked for exact allowlist membership and correspondence to cited claims, not network availability or freshness.

| Scenario | Actual status | Assessment |
| --- | --- | --- |
| ITC pricing | answered | Six explicitly named rows returned USD 18,995 with the approved ITC price-list source. Unlabelled row omitted. |
| Mexico vs Turkey | unknown | Appropriate: no comparable cancer-treatment price evidence for both countries. Other therapy prices cannot establish this comparison. |
| Adrenal package inclusions | answered | Ten claims backed by exact quotes from the approved adrenal package page; matching citation and medical disclaimer. |
| Recovery follow-up | unknown | Prior adrenal-package context was used. Source distinguishes three-week treatment and three-month follow-up, neither a recovery duration. |
| Appointment booking | restricted | Declines to book; directs user to PlacidWay. |
| Weather | restricted | Declines out-of-scope request. |
| Diabetes suitability | restricted | Declines personalized medical advice and refers to a doctor. |
| Misleading $50 | unknown | Does not confirm a treatment price; internal model attempt to label an initial-consultation component as USD was rejected. |

This is source-fidelity assessment, not independent medical validation. Eight answers remain pending independent human approval. Real-browser link rendering, live-page freshness, hosted security, concurrent load, backups, and sustained provider-failure behavior remain unverified. One run per scenario is not a reliability guarantee for nondeterministic model outputs.

## Prioritized fixes

### P0 — enforce grounding before public medical use

1. **Validate semantic support and pricing qualifiers.** `validate_grounded_answer` currently verifies quote existence and monetary subsets, not whether the quote entails the whole claim. An unrelated cure claim paired with a genuine consultation quote passes. A final fixed-price claim can pass against a source saying "starts from" and "price varies". Add explicit qualifier checks and a bounded semantic support step; fail closed on uncertain medical claims. Keep source assertions attributed to the clinic, especially efficacy and marketing language. See the strict expected failures in `tests/test_assessment_safety.py`.
2. **Replace the treatment-name allowlist with structured associations.** `price_association_supported` recognizes selected English cancer phrases. Dental implants can reuse a breast-cancer price without rejection. Validate all named treatments against individual source rows or structured extracted records; preserve qualifiers, unit, package duration and provider. The successful direct table path is deliberately limited to the ITC price-list question and does not solve general association validation.
3. **Resolve currencies conservatively.** `normalize_money` assigns USD whenever USD appears in context, including mixed USD/MXN contexts. Require column/row-local currency evidence and reject ambiguous mixed-currency symbols. Do not let an unrelated USD mention resolve every `$` price in a passage.

### P0 — API security and deployment exposure

4. **Protect credentials and logs.** `.gitignore` excludes `.env` and Streamlit secrets, which is a useful baseline. Use server-side deployment secrets, validate credentials for whitespace without printing values, and rotate credentials previously exposed in terminal tracebacks. The current broad `logger.exception` can include provider exception text and headers; redact secrets before logging. Public responses already hide internal errors. No `.env` values were read during this review.
5. **Add access and cost controls before public hosting.** No authentication, per-user rate limiting, spend limits, or bounded session history is implemented in `app.py`. Configure TLS, allowed origins, trusted proxy behavior and network policy on the actual host; these were not inspected in a hosted environment. Define consent/retention rules for medical questions and limit logs containing patient information. Verify provider data-handling settings before sending sensitive history.

### P1 — performance and caching

6. **Cache embeddings and vector-store resources.** Each `retrieve_documents` call runs `load_vector_store`, which constructs embeddings again. Five scenario requests visibly reloaded weights. Cold pricing took 12.565 seconds; later factual/unknown requests took 2.399–5.983 seconds. Cache the embedding instance and loaded store per process, keyed by embedding model and active index ID; invalidate on index activation. Test concurrency and thread safety. Do not share user-specific answer/history caches across sessions.
7. **Bound generation work and conversational state.** The model receives up to eight passages and three prior user questions. `app.py` stores unbounded messages; short standalone queries can inherit the previous question through `build_search_query`. Add message/token limits and targeted follow-up detection, then measure p50/p95 latency under load. Keep deterministic table answers for supported list intents.

### P1 — ChromaDB persistence and provenance

8. **Mount durable storage and validate the active index at startup.** Current code persists Chroma, uses unique index directories, validates path traversal, disables accidental collection creation, checks model identity, and atomically replaces the manifest. Preserve these protections. The actual manifest names seven sources and 198 chunks with model `sentence-transformers/all-MiniLM-L6-v2`. Verify document counts, source hashes and ingestion/embedding configuration at startup; reject stale or incompatible indexes. The loader currently does not enforce all these manifest fields.
9. **Back up and manage indexes.** Keep the manifest and selected index together on a persistent writable volume; test restore and process restarts. Add an indexing lock and bounded cleanup of old indexes. Do not assume local embedded Chroma is safe for shared-volume multi-instance writers; select a supported service topology if scaling horizontally. Current deployment and restore behavior were not exercised.

### P1 — Groq reliability and operational visibility

10. **Handle provider failures by category.** `get_llm` sets timeout=30 and max_retries=2; outer handling returns a generic error for every exception. Add bounded backoff for rate limits/transient failures, avoid retrying authentication/configuration errors, track total request deadlines and token usage, and test timeouts/429/5xx/structured-output failures with provider-shaped fixtures. One Groq connectivity test passed; the new synthetic-error test checks public error sanitization, not sustained provider resilience.
11. **Make validation outcomes observable safely.** Current rejection logs are INFO-level and may be hidden by default Streamlit logging. Emit structured reason counters and request IDs without credentials, full prompts, or sensitive history. Monitor unknown/error rates separately. Distinguish model-unanswerable, quote mismatch, amount mismatch, and retrieval-empty outcomes.

### P2 — repeatable deployment and maintenance

12. **Pin and lock dependencies; document deployment.** `requirements.txt` is unpinned and no Docker/deployment configuration or lockfile was found in the inspected project scope. Build a reproducible supported-Python image, prefetch the embedding model, separate indexing from web startup, and add a readiness check for source snapshots and active Chroma index. Resolve duplicate config assignments in `src/config.py` carefully (the later values currently win).
13. **Separate offline and integration tests.** The current default pytest suite includes live Hugging Face and Groq calls, with weak assertions in their connection scripts. Mark integration tests, strengthen assertions, run deterministic safety tests on each change, and run the eight live scenarios against an identified source snapshot before deployment. Require all four known failing safeguard tests to pass before declaring production grounding complete.

## Reproduce

```powershell
python -m pytest -q --junitxml=test_results/pytest_results.xml
python -m tests.evaluate_chatbot
```

The saved source-review notes apply to the observed assessment run; a new run requires renewed source/semantic review. `python -m tests.review_assessment` attaches checks and the recorded review observations to the saved run; it does not regenerate answers or provide human approval.

The codebase graph MCP returned `Transport closed`; this review used direct source inspection. Browser behavior and external page freshness were not inferred from unit test success.
