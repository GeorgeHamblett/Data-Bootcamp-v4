"""EPO Open Patent Services integration with safe CQL query construction.

The EPO connector is deliberately conservative:
- patent identifiers are searched with pn=
- non-patent searches require patent-relevant concepts
- title/abstract terms are always safely quoted
- CQL rejection for exploratory non-ID searches is treated as no relevant EPO
  match rather than a hard app failure
"""

from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

from settings import Settings, is_missing_credential
from similarity.identifiers import (
    is_nihr_identifier,
    is_patent_identifier,
    is_trial_identifier,
    normalise_identifier,
    patent_match_key,
)
from similarity.query_builder import concept_class, is_generic_term, normalise


EPO_SOURCE = "EPO OPS"
EPO_SEARCH_PATH = "/rest-services/published-data/search/biblio"


EPO_GENERIC_TERMS = {
    "the",
    "a",
    "an",
    "early",
    "consistent",
    "adults",
    "patients",
    "people",
    "community",
    "support",
    "detection",
    "device",
    "platform",
    "system",
    "study",
    "research",
    "application",
    "proposal",
    "funding",
    "grant",
    "wounds",
    "wound",
    "care",
    "nhs",
    "medical device",
    "digital tool",
    "impact model",
    "health economic model",
    "budget impact model",
    "usual care",
    "endpoint",
    "endpoints",
    "comparator",
    "trial",
    "pilot",
    "feasibility",
    "acceptability",
    "usability",
    "ukca",
    "dtac",
    "iso 13485",
    "iso 14971",
}


def _safe_result(
    status: str,
    message: str,
    *,
    error_type: str = "",
    http_status: int | None = None,
    query_terms_used: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": EPO_SOURCE,
        "status": status,
        "matches_found": 0,
        "raw_records_returned": 0,
        "top_match": message,
        "raw": {},
        "all_candidates": [],
        "score": 0.0,
        "risk": "NONE",
        "matched_concepts": [],
        "why_relevant": message,
        "link_or_id": "",
        "query_terms_used": query_terms_used or [],
    }

    if error_type:
        result["error_type"] = error_type

    if http_status is not None:
        result["http_status"] = http_status

    return result


def _empty_success(
    message: str = "No relevant EPO match found",
    *,
    query_terms_used: list[str] | None = None,
) -> dict[str, Any]:
    return _safe_result(
        "success",
        message,
        query_terms_used=query_terms_used or [],
    )


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _token(settings: Settings) -> str:
    import requests

    creds = f"{settings.epo_ops_consumer_key}:{settings.epo_ops_consumer_secret}".encode()

    headers = {
        "Authorization": "Basic " + base64.b64encode(creds).decode(),
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(
        settings.epo_ops_auth_url,
        data={"grant_type": "client_credentials"},
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    return str(response.json()["access_token"])


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
        term = _clean_epo_term(raw)
        key = normalise(term)

        if key and key not in seen:
            seen.add(key)
            out.append(term)

    return out


def _clean_epo_term(term: str) -> str:
    cleaned = str(term or "").strip()

    if is_patent_identifier(cleaned):
        return normalise_identifier(cleaned)

    cleaned = cleaned.replace("…", " ")
    cleaned = re.sub(r"\.\.\.+", " ", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9 +#_/\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:,/-")

    return cleaned


def _patent_number_for_cql(term: str) -> str:
    return patent_match_key(term)


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
    ]

    return any(re.search(pattern, key) for pattern in bad_patterns)


def _is_safe_epo_term(term: str) -> bool:
    cleaned = _clean_epo_term(term)
    key = normalise(cleaned)

    if not cleaned or not key:
        return False

    if is_patent_identifier(cleaned):
        return True

    # Correct source routing.
    if is_nihr_identifier(cleaned) or is_trial_identifier(cleaned):
        return False

    if is_generic_term(cleaned):
        return False

    if key in EPO_GENERIC_TERMS or key.rstrip("s") in EPO_GENERIC_TERMS:
        return False

    if _looks_like_bad_fragment(cleaned):
        return False

    if len(cleaned) > 80 or len(cleaned.split()) > 7:
        return False

    if re.search(r"\baged\s*\d+", key):
        return False

    if key in {
        "older adults",
        "adults",
        "patients",
        "people",
        "community",
        "nhs",
        "rehabilitation",
        "detection",
        "support",
        "platform",
        "device",
        "system",
        "software",
        "medical device",
    }:
        return False

    if key.startswith("over with ") or key.startswith("under with "):
        return False

    if all(word in EPO_GENERIC_TERMS for word in key.split()):
        return False

    cls = concept_class(cleaned)

    return cls in {
        "named_entities",
        "technical_method_or_mechanism",
        "product_or_intervention_function",
        "intervention_type",
        "clinical_or_social_care_problem",
    }


def _ta(term: str) -> str:
    """Build a safe title/abstract CQL term.

    EPO OPS CQL is fussy. Hyphens, slashes and plus signs can trigger query
    rejection if left unquoted or unnormalised. Always quote title/abstract
    terms.
    """

    cleaned = _clean_epo_term(term)

    cleaned = cleaned.replace("-", " ")
    cleaned = cleaned.replace("/", " ")
    cleaned = cleaned.replace("+", " ")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:,/-")
    cleaned = cleaned.replace('"', "")

    return f'ta="{cleaned}"'


def _selected_terms(terms: list[str] | str, *, max_terms: int = 5) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    for raw in _terms_from_query(terms):
        term = _clean_epo_term(raw)
        key = normalise(term)

        if not key or key in seen:
            continue

        if not _is_safe_epo_term(term):
            continue

        selected.append(term)
        seen.add(key)

        if len(selected) >= max_terms:
            break

    return selected


def build_epo_cql_query(
    terms: list[str] | str,
    *,
    max_terms: int = 5,
    quote_first: bool = True,
) -> str:
    """Build a high-precision, domain-agnostic EPO OPS CQL query.

    Priority:
    1. exact patent IDs via pn=
    2. named product/intervention AND technical/function term
    3. two technical/function terms
    4. technical/function term AND clinical problem term

    Named-only terms are not enough for EPO because they produce noisy or
    rejected patent queries.
    """

    selected = _selected_terms(terms, max_terms=max_terms)

    if not selected:
        return ""

    patent_ids = [term for term in selected if is_patent_identifier(term)]

    if patent_ids:
        return " or ".join(
            f"pn={_patent_number_for_cql(term)}"
            for term in patent_ids[:max_terms]
        )

    named = [
        term
        for term in selected
        if concept_class(term) == "named_entities"
    ]

    technical_or_function = [
        term
        for term in selected
        if concept_class(term)
        in {
            "technical_method_or_mechanism",
            "product_or_intervention_function",
            "intervention_type",
        }
    ]

    clinical_problem = [
        term
        for term in selected
        if concept_class(term) == "clinical_or_social_care_problem"
    ]

    if named and technical_or_function:
        return f"{_ta(named[0])} and {_ta(technical_or_function[0])}"

    if len(technical_or_function) >= 2:
        return f"{_ta(technical_or_function[0])} and {_ta(technical_or_function[1])}"

    if technical_or_function and clinical_problem:
        return f"{_ta(technical_or_function[0])} and {_ta(clinical_problem[0])}"

    return ""


def _text(elem: ET.Element) -> str:
    return " ".join(
        part.strip()
        for part in elem.itertext()
        if part and part.strip()
    )


def _parse_epo_xml(text: str) -> dict[str, Any]:
    title = ""
    abstract_parts: list[str] = []
    doc_numbers: list[str] = []
    countries: list[str] = []
    kinds: list[str] = []
    applicants: list[str] = []

    try:
        root = ET.fromstring(text.encode("utf-8"))
    except Exception:
        return {}

    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        value = _text(elem)

        if not value:
            continue

        if tag == "invention-title" and not title:
            title = value

        elif tag == "abstract":
            abstract_parts.append(value)

        elif tag == "doc-number":
            doc_numbers.append(value)

        elif tag == "country":
            countries.append(value)

        elif tag == "kind":
            kinds.append(value)

        elif tag in {"applicant-name", "name"}:
            applicants.append(value)

    # Collect direct paragraph text under abstract nodes.
    for elem in root.iter():
        if elem.tag.split("}")[-1].lower() == "abstract":
            for child in elem.iter():
                if child.tag.split("}")[-1].lower() == "p":
                    value = _text(child)
                    if value:
                        abstract_parts.append(value)

    abstract = " ".join(dict.fromkeys(abstract_parts))
    applicants = list(dict.fromkeys(applicants))
    doc_numbers = list(dict.fromkeys(doc_numbers))
    countries = list(dict.fromkeys(countries))
    kinds = list(dict.fromkeys(kinds))

    return {
        "title": title,
        "abstract": abstract,
        "doc_numbers": doc_numbers,
        "applicants": applicants,
        "country": countries[0] if countries else "",
        "kind": kinds[0] if kinds else "",
        "metadata_text": " ".join(
            [
                title,
                abstract,
                " ".join(doc_numbers),
                " ".join(applicants),
            ]
        ).strip(),
    }


def _find_values(obj: Any, wanted: set[str]) -> list[str]:
    values: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            norm_key = str(key).replace("_", "-").lower()

            if norm_key in wanted:
                if isinstance(value, dict) and "$" in value:
                    values.append(str(value["$"]))

                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "$" in item:
                            values.append(str(item["$"]))
                        elif not isinstance(item, (dict, list)):
                            values.append(str(item))

                elif not isinstance(value, (dict, list)):
                    values.append(str(value))

            values.extend(_find_values(value, wanted))

    elif isinstance(obj, list):
        for item in obj:
            values.extend(_find_values(item, wanted))

    return [str(value).strip() for value in values if str(value).strip()]


def parse_epo_metadata(text: str) -> dict[str, Any]:
    """Extract title/abstract/publication/applicant metadata from EPO XML or JSON."""

    if not text:
        return {
            "title": "",
            "abstract": "",
            "doc_numbers": [],
            "applicants": [],
            "metadata_text": "",
        }

    stripped = text.strip()
    parsed: dict[str, Any] = {}

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)

            titles = _find_values(data, {"invention-title", "title"})
            abstracts = _find_values(data, {"abstract", "p"})
            doc_numbers = _find_values(data, {"doc-number", "publication-number"})
            countries = _find_values(data, {"country"})
            kinds = _find_values(data, {"kind"})
            applicants = _find_values(data, {"applicant-name", "name", "applicant"})

            parsed = {
                "title": titles[0] if titles else "",
                "abstract": " ".join(dict.fromkeys(abstracts)),
                "doc_numbers": list(dict.fromkeys(doc_numbers)),
                "applicants": list(dict.fromkeys(applicants)),
                "country": countries[0] if countries else "",
                "kind": kinds[0] if kinds else "",
            }

        except Exception:
            parsed = {}

    if not parsed:
        parsed = _parse_epo_xml(stripped)

    if not parsed:
        parsed = {
            "title": "",
            "abstract": "",
            "doc_numbers": [],
            "applicants": [],
            "metadata_text": "",
        }

    parsed["metadata_text"] = " ".join(
        [
            str(parsed.get("title", "")),
            str(parsed.get("abstract", "")),
            " ".join(parsed.get("doc_numbers", []) or []),
            str(parsed.get("country", "")),
            str(parsed.get("kind", "")),
            " ".join(parsed.get("applicants", []) or []),
        ]
    ).strip()

    return parsed


def _extract_epo_identifier(text: str) -> str:
    match = re.search(r"\b(?:EP|WO|US)\s?\d{6,}[A-Z0-9]*\b", text, re.I)
    return match.group(0).replace(" ", "").upper() if match else ""


def _identifier_from_metadata(metadata: dict[str, Any], text: str) -> str:
    doc_numbers = [str(v) for v in metadata.get("doc_numbers", []) or [] if str(v).strip()]
    country = str(metadata.get("country", "") or "").upper()
    kind = str(metadata.get("kind", "") or "").upper()

    for doc in doc_numbers:
        compact = re.sub(r"\s+", "", doc).upper()

        if re.match(r"^(?:US|EP|WO)\d+", compact):
            if re.search(r"(?:A\d|B\d|C\d|U\d|Y\d)$", compact, re.I):
                return compact
            return compact + kind if kind else compact

        if country in {"US", "EP", "WO"} and compact.isdigit():
            return f"{country}{compact}{kind}"

    extracted = _extract_epo_identifier(text)
    if extracted:
        return extracted

    return ""


def check_epo_credentials(settings: Settings) -> dict[str, Any]:
    """Request only an OAuth token to verify EPO OPS credentials."""

    if is_missing_credential(settings.epo_ops_consumer_key) or is_missing_credential(
        settings.epo_ops_consumer_secret
    ):
        return _safe_result(
            "not_run",
            "EPO OPS credentials missing.",
            error_type="missing_credentials",
        )

    try:
        _token(settings)
        return {
            "source": EPO_SOURCE,
            "status": "success",
            "message": "EPO OPS authentication succeeded.",
            "error_type": "",
        }

    except Exception as exc:
        return _safe_result(
            "error",
            "EPO OPS authentication failed. Check consumer key and secret.",
            error_type="authentication_failed",
            http_status=_http_status(exc),
        )


def search_epo(query: str | list[str], settings: Settings) -> dict[str, Any]:
    import requests

    query_terms_used = _selected_terms(query, max_terms=5)

    if is_missing_credential(settings.epo_ops_consumer_key) or is_missing_credential(
        settings.epo_ops_consumer_secret
    ):
        return _safe_result(
            "not_run",
            "Live EPO search not run because credentials are missing.",
            error_type="missing_credentials",
            query_terms_used=query_terms_used,
        )

    try:
        token = _token(settings)

    except Exception as exc:
        return _safe_result(
            "error",
            "EPO OPS authentication failed. Check consumer key and secret.",
            error_type="authentication_failed",
            http_status=_http_status(exc),
            query_terms_used=query_terms_used,
        )

    cql = build_epo_cql_query(query)

    if not cql:
        return _safe_result(
            "not_run",
            "No safe EPO OPS query terms available after cleaning.",
            error_type="no_safe_terms",
            query_terms_used=query_terms_used,
        )

    fallback_cql = build_epo_cql_query(query, max_terms=2, quote_first=False)

    url = f"{settings.epo_ops_base_url.rstrip('/')}{EPO_SEARCH_PATH}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    candidates = [candidate for candidate in [cql, fallback_cql] if candidate]
    tried: set[str] = set()

    for idx, candidate in enumerate(candidates):
        if candidate in tried:
            continue

        tried.add(candidate)

        try:
            response = requests.get(
                url,
                params={"q": candidate, "Range": "1-5"},
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()

            text = response.text or ""
            metadata = parse_epo_metadata(text)

            title = metadata.get("title") or (
                "Patent record returned; title/abstract not parsed" if text else ""
            )

            link_or_id = _identifier_from_metadata(metadata, text)

            if not text:
                return _empty_success(query_terms_used=query_terms_used)

            return {
                "source": EPO_SOURCE,
                "status": "success",
                "matches_found": 1,
                "raw_records_returned": 1,
                "top_match": title or "Patent record returned",
                "raw": metadata
                if metadata.get("metadata_text")
                else {**metadata, "unparsed_excerpt": text[:1000]},
                "all_candidates": [],
                "link_or_id": link_or_id,
                "query_terms_used": query_terms_used,
            }

        except requests.HTTPError as exc:
            status = _http_status(exc)

            # EPO uses 404 for no results in some endpoints.
            if status == 404:
                return _empty_success(query_terms_used=query_terms_used)

            # Try fallback on CQL rejection.
            if status in {400, 422} and idx == 0 and len(candidates) > 1:
                continue

            # For exploratory non-patent searches, do not fail the whole
            # similarity checker just because EPO rejects the CQL.
            if status in {400, 422}:
                return _empty_success(
                    "No relevant EPO match found",
                    query_terms_used=query_terms_used,
                )

            return _safe_result(
                "error",
                "EPO OPS request failed. Query URL suppressed.",
                error_type="request_failed",
                http_status=status,
                query_terms_used=query_terms_used,
            )

        except Exception as exc:
            return _safe_result(
                "error",
                "EPO OPS request failed. Query URL suppressed.",
                error_type="request_failed",
                http_status=_http_status(exc),
                query_terms_used=query_terms_used,
            )

    return _empty_success(query_terms_used=query_terms_used)


# Compatibility alias in case another module imports this name.
search_epo_ops = search_epo