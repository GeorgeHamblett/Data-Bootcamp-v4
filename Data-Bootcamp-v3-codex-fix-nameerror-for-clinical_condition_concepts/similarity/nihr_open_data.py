"""NIHR Open Data / Funding Awards integration.

This connector searches:
1. NIHR Open Data / Opendatasoft records endpoint, if configured.
2. Official NIHR public pages as fallback evidence.
3. Direct NIHR Funding and Awards URLs for exact NIHR-style public IDs.

Important:
- Do not treat raw API records as relevant just because they were returned.
- Do not fail the whole connector if Open Data errors before fallback runs.
- Prioritise exact NIHR public identifiers.
- Do not send patent or trial identifiers to NIHR Open Data.
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from settings import Settings, is_missing_credential
from similarity.identifiers import (
    is_nihr_identifier,
    is_patent_identifier,
    is_trial_identifier,
    normalise_identifier,
)
from similarity.query_builder import concept_class, is_generic_term, normalise


NIHR_SOURCE = "NIHR Open Data"


DEFAULT_NIHR_PUBLIC_SOURCE_URLS = (
    "https://www.nihr.ac.uk/ai-health-and-care-awards-funded-projects-2020",
    "https://www.nihr.ac.uk/invention-innovation-pda-call-23-stage-1-minutes",
)


NIHR_BLOCKED_TERMS = {
    # Source/search terms
    "epo",
    "epo ops",
    "lens",
    "lens scholarly",
    "open data",
    "nihr open data",
    "ukipo",
    "uspto",
    "hra",
    "google patents",
    "source",
    "sources",
    "anchor",
    "anchors",
    "machine-readable",
    "machine readable",

    # Generic organisations/schemes
    "nihr",
    "nhs",
    "nhs england",
    "nice",
    "mhra",
    "rss",
    "i4i",
    "pda",
    "product development award",
    "invention",
    "invention for innovation",
    "medtech",
    "sme",
    "small and medium enterprise",
    "small and medium-sized enterprise",

    # Finance / health economics
    "health economics",
    "health economic",
    "economic evaluation",
    "economic model",
    "decision-analytic model",
    "decision tree",
    "decision-tree modelling",
    "markov model",
    "cost effectiveness",
    "cost-effectiveness analysis",
    "cost utility",
    "cost-utility analysis",
    "cea",
    "cua",
    "qaly",
    "qalys",
    "icer",
    "roi",
    "budget",
    "budget impact",
    "budget impact model",
    "impact model",
    "cost model",
    "value for money",

    # Regulatory/checklist/readiness
    "trl",
    "technology readiness",
    "technology readiness level",
    "regulatory readiness",
    "implementation readiness",
    "adoption readiness",
    "ukca",
    "ukca ready",
    "ukca-ready",
    "ukca compliant",
    "ukca-compliant",
    "ce marking",
    "iso 13485",
    "iso 14971",
    "iec 62304",
    "dtac",
    "dtac aligned",
    "dtac-aligned",
    "clinical validation",
    "clinical validation needs",
    "quality management system",
    "technical documentation",
    "technical file",
    "compliance pathway",

    # Application/document headings
    "knowledge",
    "knowledge mobilisation",
    "project management",
    "work package",
    "references",
    "reference list",
    "appendix",
    "methodology",
    "aims",
    "objectives",
    "background",
    "rationale",
    "application",
    "proposal",
    "guidance",
    "funding",
    "grant",
    "plain english summary",

    # Evaluation fragments
    "endpoint",
    "endpoints",
    "primary endpoint",
    "secondary endpoint",
    "outcome",
    "outcomes",
    "feasibility",
    "acceptability",
    "usability",
    "recruitment",
    "retention",
    "fidelity",
    "interviews",
    "comparator",
    "usual care",
    "current best practice",
    "control arm",

    # Weak standalone terms
    "patients",
    "people",
    "adults",
    "older adults",
    "children",
    "community",
    "care",
    "services",
    "clinic",
    "clinics",
    "hospital",
    "population",
    "setting",
    "support",
    "platform",
    "device",
    "system",
    "software",
    "intervention",
    "therapeutic",
    "diagnostic",
    "medical device",
    "digital tool",
}


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _safe_empty_result(
    searched_term: str = "",
    *,
    raw_records_returned: int = 0,
    all_candidates: list[dict[str, Any]] | None = None,
    query_terms_used: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source": NIHR_SOURCE,
        "status": "success",
        "matches_found": raw_records_returned,
        "raw_records_returned": raw_records_returned,
        "top_match": "No relevant NIHR Open Data match found",
        "searched_term": searched_term,
        "raw": {},
        "all_candidates": all_candidates or [],
        "link_or_id": "",
        "query_terms_used": query_terms_used or [],
        "connector_warnings": warnings or [],
    }


def _clean_term(term: str) -> str:
    cleaned = str(term or "").replace("…", " ")
    cleaned = re.sub(r"\.\.\.+", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 +#_/\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:,/-")
    return cleaned


def _looks_like_bad_fragment(term: str) -> bool:
    key = normalise(term)

    bad_patterns = [
        r"\bhealth economic model\b",
        r"\bhealth economics\b",
        r"\beconomic evaluation\b",
        r"\bbudget impact model\b",
        r"\bimpact model\b",
        r"\bcost effectiveness\b",
        r"\bcost-effectiveness\b",
        r"\bendpoint\b",
        r"\bendpoints\b",
        r"\bcomparator\b",
        r"\busual care\b",
        r"\busual nhs\b",
        r"\bnhs wound assessment\b",
        r"\btrial of the system\b",
        r"\bconduct a prospective\b",
        r"\bprospective multi[-\s]site diagnostic\b",
        r"\bacross nhs community rehabilitation\b",
        r"\btest a new wearable digital tool\b",
        r"\bdigital tool\b",
        r"\bregulatory gap analysis\b",
        r"\bukca\b",
        r"\bdtac\b",
        r"\biso\s*13485\b",
        r"\biso\s*14971\b",
        r"\bclinical validation\b",
        r"\bwork package\b",
        r"\bwp\d+\b",
        r"\bactivities specific\b",
        r"\bdata collection complete\b",
        r"\bgo no go\b",
        r"\bgo/no-go\b",
        r"\bit combines\b",
        r"\bsummary this project\b",
        r"\bthis project will test\b",
        r"\bproject will test\b",
    ]

    return any(re.search(pattern, key) for pattern in bad_patterns)


def _is_safe_nihr_term(term: str) -> bool:
    cleaned = _clean_term(term)
    key = normalise(cleaned)

    if not cleaned or not key:
        return False

    # Correct source routing.
    if is_patent_identifier(cleaned) or is_trial_identifier(cleaned):
        return False

    if is_nihr_identifier(cleaned):
        return True

    if key in NIHR_BLOCKED_TERMS or key.rstrip("s") in NIHR_BLOCKED_TERMS:
        return False

    if is_generic_term(cleaned):
        return False

    if _looks_like_bad_fragment(cleaned):
        return False

    if len(cleaned) > 90 or len(cleaned.split()) > 8:
        return False

    cls = concept_class(cleaned)

    return cls in {
        "named_entities",
        "technical_method_or_mechanism",
        "product_or_intervention_function",
        "intervention_type",
        "clinical_or_social_care_problem",
        "population_setting",
    }


def _term_rank(term: str) -> tuple[int, str]:
    if is_nihr_identifier(term):
        return (0, normalise(term))

    cls = concept_class(term)

    if cls == "named_entities":
        return (1, normalise(term))

    if cls == "technical_method_or_mechanism":
        return (2, normalise(term))

    if cls == "product_or_intervention_function":
        return (3, normalise(term))

    if cls == "intervention_type":
        return (4, normalise(term))

    if cls == "clinical_or_social_care_problem":
        return (5, normalise(term))

    if cls == "population_setting":
        return (6, normalise(term))

    return (99, normalise(term))


def _terms_from_query(query: str | list[str]) -> list[str]:
    if isinstance(query, list):
        raw_terms = [str(term) for term in query if str(term).strip()]
    else:
        quoted = re.findall(r'"([^"]+)"', str(query))
        remainder = re.sub(r'"[^"]+"', " ", str(query))
        bare = re.split(r"\s+AND\s+|\s+OR\s+|[,;]", remainder, flags=re.I)
        raw_terms = quoted + [part.strip() for part in bare if part.strip()]

    out: list[str] = []
    seen: set[str] = set()

    for raw in raw_terms:
        term = _clean_term(raw)
        key = normalise(term)

        if not key or key in seen:
            continue

        if not _is_safe_nihr_term(term):
            continue

        seen.add(key)
        out.append(term)

    return sorted(out, key=_term_rank)


def _headers(settings: Settings) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 NIHR-similarity-checker/1.0",
    }

    api_key = getattr(settings, "nihr_open_data_api_key", "")

    if not is_missing_credential(api_key):
        headers["apikey"] = api_key

    return headers


def _api_url(settings: Settings) -> str:
    base_url = str(getattr(settings, "nihr_open_data_base_url", "") or "").rstrip("/")
    dataset_id = str(getattr(settings, "nihr_open_data_dataset_id", "") or "").strip("/")

    if not base_url or not dataset_id:
        return ""

    return f"{base_url}/catalog/datasets/{dataset_id}/records"


def _flatten_text(value: Any) -> str:
    chunks: list[str] = []

    if isinstance(value, dict):
        for key, item in value.items():
            chunks.append(str(key))

            if isinstance(item, (dict, list)):
                chunks.append(_flatten_text(item))
            elif item is not None:
                chunks.append(str(item))

    elif isinstance(value, list):
        for item in value:
            chunks.append(_flatten_text(item))

    elif value is not None:
        chunks.append(str(value))

    return " ".join(chunks)


def _record_title(record: dict[str, Any]) -> str:
    for key in (
        "project_title",
        "award_title",
        "title",
        "name",
        "acronym",
        "recordid",
        "project_id",
    ):
        value = record.get(key)

        if value:
            return str(value)

    return ""


def _record_link_or_id(record: dict[str, Any], fallback_url: str = "") -> str:
    for key in (
        "funding_and_awards_link",
        "url",
        "link",
        "id",
        "recordid",
        "project_id",
        "award_id",
        "reference",
    ):
        value = record.get(key)

        if value:
            return str(value)

    return fallback_url


def _record_id(record: dict[str, Any]) -> str:
    for key in (
        "project_id",
        "award_id",
        "recordid",
        "id",
        "reference",
        "funding_and_awards_link",
        "url",
    ):
        value = record.get(key)

        if value:
            return str(value)

    return normalise(_record_title(record))[:100]


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    for wrapper_key in ("fields", "record"):
        wrapped = record.get(wrapper_key)

        if isinstance(wrapped, dict):
            return {**record, **wrapped}

    return record


def _extract_results(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []

    for key in ("results", "records", "data"):
        results = data.get(key)

        if isinstance(results, list):
            return [_normalise_record(item) for item in results if isinstance(item, dict)]

    return []


def _contains_phrase(haystack: str, phrase: str) -> bool:
    key = normalise(phrase)

    if not key:
        return False

    if len(key) <= 6 and " " not in key:
        return bool(re.search(rf"\b{re.escape(key)}\b", haystack))

    return key in haystack or key.replace(" ", "-") in haystack


def _candidate_score(record: dict[str, Any], searched_term: str, all_terms: list[str]) -> float:
    text = normalise(_flatten_text(record))
    searched_key = normalise(searched_term)

    if not text:
        return 0.0

    # Exact NIHR ID match is strongest.
    for term in all_terms:
        if is_nihr_identifier(term) and normalise(normalise_identifier(term)) in text:
            return 100.0

    if is_nihr_identifier(searched_term) and searched_key in text:
        return 100.0

    named_hits = [
        term
        for term in all_terms
        if concept_class(term) == "named_entities" and _contains_phrase(text, term)
    ]

    method_or_function_hits = [
        term
        for term in all_terms
        if concept_class(term)
        in {
            "technical_method_or_mechanism",
            "product_or_intervention_function",
            "intervention_type",
        }
        and _contains_phrase(text, term)
    ]

    problem_hits = [
        term
        for term in all_terms
        if concept_class(term) == "clinical_or_social_care_problem"
        and _contains_phrase(text, term)
    ]

    cls = concept_class(searched_term)

    if cls == "named_entities" and _contains_phrase(text, searched_term):
        if method_or_function_hits or problem_hits:
            return 90.0
        return 65.0

    if named_hits and method_or_function_hits:
        return 90.0

    if named_hits and problem_hits:
        return 80.0

    if method_or_function_hits and problem_hits:
        return 70.0

    if method_or_function_hits:
        return 40.0

    if problem_hits:
        return 20.0

    unrelated_signals = {
        "gene targeted transgenic mice",
        "prp",
        "prion",
        "cjd",
        "tse",
        "murine",
    }

    if any(signal in text for signal in unrelated_signals):
        return 0.0

    return 0.0


def _param_sets_for_term(term: str) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = [
        {"limit": 5, "search": term},
    ]

    if is_nihr_identifier(term):
        safe = normalise_identifier(term)
        params.extend(
            [
                {"limit": 5, "where": f'project_id="{safe}"'},
                {"limit": 5, "where": f'recordid="{safe}"'},
                {"limit": 5, "where": f'award_id="{safe}"'},
            ]
        )

    return params


def _search_open_data_once(
    term: str,
    settings: Settings,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    import requests

    url = _api_url(settings)

    if not url:
        return 0, [], ["NIHR Open Data endpoint is not configured."]

    headers = _headers(settings)
    all_records: list[dict[str, Any]] = []
    raw_count = 0
    warnings: list[str] = []

    for params in _param_sets_for_term(term):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()

            data = response.json()
            records = _extract_results(data)
            raw_count += len(records)

            if records:
                all_records.extend(records)

        except Exception as exc:
            status = _http_status(exc)
            warnings.append(
                f"Open Data subquery failed for term '{term}'"
                + (f" with HTTP {status}" if status else "")
            )
            continue

    return raw_count, all_records, warnings


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _html_title(html: str) -> str:
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)

    if title:
        return re.sub(r"\s+", " ", unescape(title.group(1))).strip()

    h1 = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)

    if h1:
        return re.sub(
            r"\s+",
            " ",
            unescape(re.sub(r"<[^>]+>", " ", h1.group(1))),
        ).strip()

    return ""


def _snippet_around(text: str, term: str, window: int = 350) -> str:
    key_text = text.lower()
    key_term = term.lower()
    idx = key_text.find(key_term)

    if idx < 0:
        return text[: window * 2]

    start = max(0, idx - window)
    end = min(len(text), idx + len(term) + window)

    return text[start:end].strip()


def _public_urls_from_settings(settings: Settings) -> list[str]:
    urls: list[str] = list(DEFAULT_NIHR_PUBLIC_SOURCE_URLS)

    for attr in (
        "nihr_public_source_urls",
        "nihr_similarity_source_urls",
        "similarity_source_urls",
        "external_similarity_source_urls",
    ):
        value = getattr(settings, attr, None)

        if not value:
            continue

        if isinstance(value, str):
            urls.extend(
                part.strip()
                for part in re.split(r"[\n,;]", value)
                if part.strip()
            )

        elif isinstance(value, list):
            urls.extend(str(item).strip() for item in value if str(item).strip())

    seen: set[str] = set()
    out: list[str] = []

    for url in urls:
        if url not in seen:
            seen.add(url)
            out.append(url)

    return out


def _direct_funding_awards_url(term: str) -> str:
    if not is_nihr_identifier(term):
        return ""

    return f"https://fundingawards.nihr.ac.uk/award/{normalise_identifier(term)}"


def _candidate_from_public_page(
    *,
    url: str,
    html: str,
    term: str,
    all_terms: list[str],
    source_kind: str,
) -> dict[str, Any] | None:
    text = _html_to_text(html)
    title = _html_title(html)

    searchable = normalise(f"{title} {text} {url}")

    exact_url_hit = (
        is_nihr_identifier(term)
        and normalise_identifier(term).lower() in url.lower()
    )

    term_hit = _contains_phrase(searchable, term)

    supporting_hits = [
        item
        for item in all_terms
        if item != term and _contains_phrase(searchable, item)
    ]

    if not term_hit and not exact_url_hit and not supporting_hits:
        return None

    snippet = _snippet_around(text, term) if term_hit else text[:700]

    if not title:
        if source_kind == "direct_award" and is_nihr_identifier(term):
            title = f"NIHR Funding and Awards record: {normalise_identifier(term)}"
        else:
            title = "NIHR public source match"

    project_id = normalise_identifier(term) if is_nihr_identifier(term) else ""

    record = {
        "project_id": project_id,
        "award_id": project_id,
        "project_title": title,
        "title": title,
        "snippet": snippet,
        "description": snippet,
        "metadata_text": " ".join(
            [
                title,
                project_id,
                snippet,
                url,
                " ".join(supporting_hits),
            ]
        ).strip(),
        "funding_and_awards_link": url,
        "url": url,
        "_source_kind": source_kind,
        "_searched_term": term,
    }

    score = _candidate_score(record, term, all_terms)

    if exact_url_hit:
        score = max(score, 100.0)

    record["_preselection_score"] = score

    return record


def _search_public_pages(
    terms: list[str],
    settings: Settings,
) -> tuple[list[dict[str, Any]], list[str]]:
    import requests

    headers = _headers(settings)
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    exact_terms = [term for term in terms if is_nihr_identifier(term)]

    # 1. Direct Funding and Awards URLs for exact NIHR IDs.
    for term in exact_terms:
        url = _direct_funding_awards_url(term)

        if not url:
            continue

        try:
            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code >= 400:
                warnings.append(f"Direct NIHR award URL failed for {term} with HTTP {response.status_code}")
                continue

            candidate = _candidate_from_public_page(
                url=url,
                html=response.text or "",
                term=term,
                all_terms=terms,
                source_kind="direct_award",
            )

            if candidate:
                candidates.append(candidate)

        except Exception as exc:
            status = _http_status(exc)
            warnings.append(
                f"Direct NIHR award URL failed for {term}"
                + (f" with HTTP {status}" if status else "")
            )
            continue

    # 2. NIHR public pages such as award lists and committee minutes.
    page_terms = exact_terms + [
        term for term in terms if concept_class(term) == "named_entities"
    ]

    for url in _public_urls_from_settings(settings):
        try:
            response = requests.get(url, headers=headers, timeout=20)

            if response.status_code >= 400:
                warnings.append(f"NIHR public page failed with HTTP {response.status_code}: {url}")
                continue

            html = response.text or ""

        except Exception as exc:
            status = _http_status(exc)
            warnings.append(
                f"NIHR public page request failed: {url}"
                + (f" with HTTP {status}" if status else "")
            )
            continue

        for term in page_terms[:8]:
            candidate = _candidate_from_public_page(
                url=url,
                html=html,
                term=term,
                all_terms=terms,
                source_kind="nihr_public_page",
            )

            if candidate:
                candidates.append(candidate)

    return candidates, warnings


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for candidate in candidates:
        key = (
            _record_id(candidate)
            or _record_link_or_id(candidate)
            or normalise(_record_title(candidate))
        )

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(candidate)

    return out


def _build_result(
    *,
    raw_records_returned: int,
    top: dict[str, Any],
    searched_term: str,
    all_candidates: list[dict[str, Any]],
    query_terms_used: list[str],
    fallback_url: str,
    warnings: list[str],
) -> dict[str, Any]:
    if not top:
        return _safe_empty_result(
            searched_term,
            raw_records_returned=raw_records_returned,
            all_candidates=all_candidates[:10],
            query_terms_used=query_terms_used,
            warnings=warnings,
        )

    return {
        "source": NIHR_SOURCE,
        "status": "success",
        "matches_found": raw_records_returned,
        "raw_records_returned": raw_records_returned,
        "top_match": _record_title(top),
        "searched_term": searched_term,
        "raw": top,
        "all_candidates": all_candidates[:10],
        "link_or_id": _record_link_or_id(top, fallback_url),
        "query_terms_used": query_terms_used,
        "connector_warnings": warnings,
    }


def search_nihr_open_data(query: str | list[str], settings: Settings) -> dict[str, Any]:
    terms = _terms_from_query(query)

    if not terms:
        return _safe_empty_result("")

    raw_records_returned = 0
    all_candidates: list[dict[str, Any]] = []
    best_record: dict[str, Any] = {}
    best_score = 0.0
    best_term = terms[0]
    warnings: list[str] = []

    # 1. Search NIHR Open Data endpoint.
    # Never return early on endpoint failure; public-page fallback must still run.
    for term in terms[:8]:
        raw_count, records, sub_warnings = _search_open_data_once(term, settings)
        warnings.extend(sub_warnings)
        raw_records_returned += raw_count

        for record in records:
            candidate = {
                **record,
                "_searched_term": term,
                "_preselection_score": _candidate_score(record, term, terms),
            }

            all_candidates.append(candidate)

            score = float(candidate.get("_preselection_score", 0.0))

            if score > best_score:
                best_score = score
                best_record = candidate
                best_term = term

        if best_score >= 100.0:
            break

    # 2. Always try official NIHR public fallback pages.
    public_candidates, public_warnings = _search_public_pages(terms, settings)
    warnings.extend(public_warnings)
    raw_records_returned += len(public_candidates)

    for candidate in public_candidates:
        all_candidates.append(candidate)

        score = float(candidate.get("_preselection_score", 0.0))

        if score > best_score:
            best_score = score
            best_record = candidate
            best_term = str(candidate.get("_searched_term") or best_term)

    all_candidates = _dedupe_candidates(all_candidates)
    all_candidates.sort(
        key=lambda item: float(item.get("_preselection_score", 0.0)),
        reverse=True,
    )

    fallback_url = _api_url(settings) or "https://fundingawards.nihr.ac.uk/"

    if best_score <= 0.0:
        return _build_result(
            raw_records_returned=raw_records_returned,
            top={},
            searched_term=best_term,
            all_candidates=all_candidates,
            query_terms_used=terms,
            fallback_url=fallback_url,
            warnings=warnings,
        )

    return _build_result(
        raw_records_returned=raw_records_returned,
        top=best_record,
        searched_term=best_term,
        all_candidates=all_candidates,
        query_terms_used=terms,
        fallback_url=fallback_url,
        warnings=warnings,
    )