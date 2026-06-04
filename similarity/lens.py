"""Lens Scholarly API integration.

This connector performs a cautious scholarly search using source-specific safe
terms prepared by service.py.

Important:
- Lens raw records are not automatically relevant matches.
- This connector normalises Lens results and chooses the best plausible candidate
  using lightweight phrase overlap only.
- Final relevance scoring is handled by similarity.scoring.
- Patent IDs should not normally be sent to Lens.
- Generic source/checklist/finance/regulatory terms should not be sent.
"""

from __future__ import annotations

import re
from typing import Any

from settings import Settings, is_missing_credential
from similarity.identifiers import (
    is_nihr_identifier,
    is_patent_identifier,
    is_trial_identifier,
)
from similarity.query_builder import concept_class, is_generic_term, normalise


LENS_SOURCE = "Lens Scholarly"


LENS_BLOCKED_TERMS = {
    # Source/search words
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

    # Generic organisations/schemes/sector labels
    "nihr",
    "nhs",
    "nhs england",
    "nice",
    "mhra",
    "rss",
    "medtech",
    "sme",
    "small and medium enterprise",
    "small and medium-sized enterprise",
    "i4i",
    "pda",
    "product development award",
    "invention",
    "invention for innovation",

    # Finance/checklist/evaluation
    "health economics",
    "economic evaluation",
    "economic model",
    "decision-analytic model",
    "budget",
    "budget impact",
    "cost model",
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
    "eq-5d",
    "eq-5d-5l",
    "pss",
    "sus",
    "pssuq",
    "endpoint",
    "endpoints",
    "outcome",
    "outcomes",
    "feasibility",
    "acceptability",
    "usability",
    "recruitment",
    "retention",
    "fidelity",
    "interviews",

    # Regulatory/readiness
    "trl",
    "technology readiness",
    "technology readiness level",
    "regulatory readiness",
    "implementation readiness",
    "adoption readiness",
    "ukca",
    "ukca ready",
    "ukca-ready",
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

    # Weak standalone terms
    "patients",
    "people",
    "adults",
    "older adults",
    "community",
    "services",
    "support",
    "platform",
    "device",
    "software",
    "system",
}


def _safe_empty_result(
    *,
    status: str = "success",
    message: str = "No relevant match found",
    raw_records_returned: int = 0,
    query_terms_used: list[str] | None = None,
    all_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "source": LENS_SOURCE,
        "status": status,
        "matches_found": raw_records_returned,
        "raw_records_returned": raw_records_returned,
        "top_match": message,
        "raw": {},
        "all_candidates": all_candidates or [],
        "link_or_id": "",
        "query_terms_used": query_terms_used or [],
    }


def _error_result(
    message: str,
    *,
    error_type: str = "",
    http_status: int | None = None,
    query_terms_used: list[str] | None = None,
) -> dict[str, Any]:
    result = _safe_empty_result(
        status="error",
        message=message,
        raw_records_returned=0,
        query_terms_used=query_terms_used,
    )

    if error_type:
        result["error_type"] = error_type

    if http_status is not None:
        result["http_status"] = http_status

    return result


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _clean_term(term: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 +#_/\-]", " ", str(term or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:,/-")
    return cleaned


def _looks_like_outcome_scale(term: str) -> bool:
    key = normalise(term)

    if key in {"sus", "pss", "pssuq", "eq-5d", "eq-5d-5l"}:
        return True

    if "qol" in key:
        return True

    if key.endswith(" scale") or key.endswith(" questionnaire") or key.endswith(" inventory"):
        return True

    return False


def _looks_like_sentence_fragment(term: str) -> bool:
    key = normalise(term)
    words = key.split()

    if is_trial_identifier(term):
        return False

    bad_patterns = [
        r"\baccess to sufficiently\b",
        r"\bsufficiently intensive support\b",
        r"\bdata collection complete\b",
        r"\bcomplete for algorithm\b",
        r"\bclinical validation needs\b",
        r"\btheir risk of\b",
        r"\breduce their risk\b",
        r"\bconsistent detection of\b",
        r"\bearly and consistent\b",
        r"\bconsume significant\b",
        r"\bsubstantial patient burden\b",
        r"\bis a translational\b",
        r"\bbetter use of\b",
        r"\bworkforce capacity\b",
        r"\bcommunity nursing capacity\b",
        r"\bsupports? nhs adoption\b",
        r"\bcreating substantial pressure\b",
    ]

    if any(re.search(pattern, key) for pattern in bad_patterns):
        return True

    if len(words) > 8:
        return True

    if re.search(
        r"\b(is|are|was|were|will|would|could|should|consume|consuming|create|creating|reduce|reducing|improve|improving|supporting|needs|needed|designed|including|capacity|burden|pressure|adoption|implementation|complete|confirmed|aligned)\b",
        key,
    ):
        return True

    return False


def _is_safe_lens_term(term: str) -> bool:
    cleaned = _clean_term(term)
    key = normalise(cleaned)

    if not cleaned or not key:
        return False

    # Correct source routing: Lens is not the patent/NIHR-award lookup source.
    if is_patent_identifier(cleaned) or is_nihr_identifier(cleaned):
        return False

    if is_trial_identifier(cleaned):
        return True

    if key in LENS_BLOCKED_TERMS or key.rstrip("s") in LENS_BLOCKED_TERMS:
        return False

    if is_generic_term(cleaned):
        return False

    if _looks_like_outcome_scale(cleaned):
        return False

    if _looks_like_sentence_fragment(cleaned):
        return False

    if re.fullmatch(r"wp\d+", key):
        return False

    if re.fullmatch(r"trl(?:\s+\d+)?(?:\s+to\s+trl?\s*\d+)?", key):
        return False

    if re.fullmatch(r"(?:ukca|ce)[-\s]?(?:ready|compliant|marked|certified)", key):
        return False

    if key.endswith(" aligned") or key.endswith(" specific") or key.endswith(" ready"):
        return False

    if len(cleaned) > 90 or len(cleaned.split()) > 8:
        return False

    if len(cleaned) < 3:
        return False

    cls = concept_class(cleaned)

    return cls in {
        "exact_identifier",
        "named_entities",
        "intervention_type",
        "technical_method_or_mechanism",
        "product_or_intervention_function",
        "clinical_or_social_care_problem",
    }


def _term_rank(term: str) -> tuple[int, str]:
    if is_trial_identifier(term):
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

    return (99, normalise(term))


def _terms_from_query(query: str | list[str]) -> list[str]:
    if isinstance(query, list):
        raw_terms = [str(term) for term in query if str(term).strip()]
    else:
        quoted = re.findall(r'"([^"]+)"', str(query))
        remainder = re.sub(r'"[^"]+"', " ", str(query))
        bare = re.split(r"\s+AND\s+|\s+OR\s+|[,;]", remainder, flags=re.I)
        raw_terms = quoted + [part.strip() for part in bare if part.strip()]

    seen: set[str] = set()
    out: list[str] = []

    for raw in raw_terms:
        term = _clean_term(raw)
        key = normalise(term)

        if not key or key in seen:
            continue

        if not _is_safe_lens_term(term):
            continue

        seen.add(key)
        out.append(term)

    return sorted(out, key=_term_rank)


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term else term


def _build_lens_query(terms: list[str]) -> str:
    # Use OR to avoid over-constraining scholarly search, but keep only
    # source-routed safe terms.
    return " OR ".join(_quote(term) for term in terms[:6])


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


def _extract_results(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []

    for key in ("data", "results", "records"):
        results = data.get(key)

        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]

    return []


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)

        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
                if isinstance(item, dict):
                    nested = _flatten_text(item).strip()
                    if nested:
                        return nested

        elif isinstance(value, dict):
            nested = _flatten_text(value).strip()
            if nested:
                return nested

        elif value:
            return str(value).strip()

    return ""


def _record_title(record: dict[str, Any]) -> str:
    return _first_value(
        record,
        (
            "title",
            "display_name",
            "article_title",
            "publication_title",
            "name",
        ),
    )


def _record_abstract(record: dict[str, Any]) -> str:
    return _first_value(
        record,
        (
            "abstract",
            "abstract_text",
            "description",
            "snippet",
            "summary",
        ),
    )


def _record_id(record: dict[str, Any]) -> str:
    value = _first_value(
        record,
        (
            "lens_id",
            "id",
            "doi",
            "external_ids",
            "pmid",
            "pmcid",
        ),
    )

    return value or normalise(_record_title(record))[:100]


def _record_link_or_id(record: dict[str, Any]) -> str:
    return _first_value(
        record,
        (
            "lens_id",
            "doi",
            "id",
            "url",
            "link",
            "external_ids",
            "pmid",
            "pmcid",
        ),
    )


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    title = _record_title(record)
    abstract = _record_abstract(record)
    link_or_id = _record_link_or_id(record)

    return {
        **record,
        "title": title,
        "abstract": abstract,
        "metadata_text": " ".join(
            [
                title,
                abstract,
                _flatten_text(record.get("external_ids", "")),
                _flatten_text(record.get("authors", "")),
                _flatten_text(record.get("source", "")),
            ]
        ).strip(),
        "_link_or_id": link_or_id,
    }


def _contains_phrase(haystack: str, phrase: str) -> bool:
    key = normalise(phrase)

    if not key:
        return False

    if len(key) <= 6 and " " not in key:
        return bool(re.search(rf"\b{re.escape(key)}\b", haystack))

    return key in haystack or key.replace(" ", "-") in haystack


def _candidate_score(record: dict[str, Any], terms: list[str]) -> float:
    """Lightweight pre-selection only.

    Final risk scoring happens later in scoring.py.
    """

    text = normalise(record.get("metadata_text") or _flatten_text(record))
    title = normalise(record.get("title") or "")

    if not text:
        return 0.0

    # Do not select obviously unrelated records based only on short acronym hits.
    exact_short_acronym_hits = [
        term
        for term in terms
        if re.fullmatch(r"[A-Z0-9]{2,6}", term)
        and _contains_phrase(text, term)
    ]

    non_acronym_hits = [
        term
        for term in terms
        if not re.fullmatch(r"[A-Z0-9]{2,6}", term)
        and _contains_phrase(text, term)
    ]

    if exact_short_acronym_hits and not non_acronym_hits:
        return 0.0

    named_hits = [
        term
        for term in terms
        if concept_class(term) == "named_entities" and _contains_phrase(text, term)
    ]

    method_or_function_hits = [
        term
        for term in terms
        if concept_class(term)
        in {
            "intervention_type",
            "product_or_intervention_function",
            "technical_method_or_mechanism",
        }
        and _contains_phrase(text, term)
    ]

    problem_hits = [
        term
        for term in terms
        if concept_class(term) == "clinical_or_social_care_problem"
        and _contains_phrase(text, term)
    ]

    if named_hits and method_or_function_hits:
        return 90.0

    if method_or_function_hits and problem_hits:
        return 75.0

    if len(method_or_function_hits) >= 2:
        return 70.0

    if named_hits and problem_hits:
        return 60.0

    if named_hits:
        return 40.0

    if method_or_function_hits:
        return 30.0

    if problem_hits:
        return 10.0

    return 0.0


def search_lens(query: str | list[str], settings: Settings) -> dict[str, Any]:
    import requests

    terms = _terms_from_query(query)

    if not terms:
        return _safe_empty_result(
            message="No safe Lens query terms available after cleaning.",
            raw_records_returned=0,
            query_terms_used=[],
        )

    if is_missing_credential(settings.lens_api_token):
        return _safe_empty_result(
            status="not_run",
            message="Lens API token missing",
            raw_records_returned=0,
            query_terms_used=terms,
        )

    headers = {
        "Authorization": f"Bearer {settings.lens_api_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": _build_lens_query(terms),
        "size": 10,
    }

    try:
        response = requests.post(
            settings.lens_api_base_url,
            json=payload,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        return _error_result(
            "Lens Scholarly request failed. Query URL suppressed.",
            error_type="request_failed",
            http_status=_http_status(exc),
            query_terms_used=terms,
        )

    raw_results = _extract_results(data)
    normalised = [_normalise_record(record) for record in raw_results]
    raw_records_returned = len(normalised)

    if not normalised:
        return _safe_empty_result(
            message="No relevant match found",
            raw_records_returned=0,
            query_terms_used=terms,
            all_candidates=[],
        )

    scored_candidates: list[dict[str, Any]] = []

    for record in normalised:
        score = _candidate_score(record, terms)
        scored_candidates.append(
            {
                **record,
                "_preselection_score": score,
            }
        )

    scored_candidates.sort(
        key=lambda item: float(item.get("_preselection_score", 0.0)),
        reverse=True,
    )

    best = scored_candidates[0]

    # Return the best raw candidate for final scoring, even if preselection is
    # low. The scoring layer will set matches_found to zero if there is no
    # meaningful overlap.
    return {
        "source": LENS_SOURCE,
        "status": "success",
        "matches_found": raw_records_returned,
        "raw_records_returned": raw_records_returned,
        "top_match": best.get("title", ""),
        "raw": best,
        "all_candidates": scored_candidates[:10],
        "link_or_id": best.get("_link_or_id", ""),
        "query_terms_used": terms,
    }