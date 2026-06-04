"""Similarity service orchestration with strict privacy gates.

This module routes already-cleaned similarity terms to source-specific APIs.

It must stay domain-agnostic:
- no wound-specific gates
- no AI-specific gates
- no device-specific gates
- no funding-stream-specific gates

EPO OPS should run when there is:
1. a patent identifier, OR
2. a named intervention/product plus a technical/function term, OR
3. at least two technical/function terms.

EPO OPS should not require a patent identifier.
EPO OPS should not run on only checklist, finance, population, setting or broad
condition terms.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from schemas import ApplicationFacts
from settings import Settings
from similarity.query_builder import (
    build_similarity_query,
    concept_class,
    is_generic_term,
    normalise,
)
from similarity.identifiers import (
    is_any_identifier,
    is_nihr_identifier as _id_is_nihr_identifier,
    is_patent_identifier as _id_is_patent_identifier,
    is_trial_identifier as _id_is_trial_identifier,
)
from similarity.scoring import score_result
from similarity.concepts import metadata_text
from similarity.lens import search_lens
from similarity.epo_ops import search_epo
from similarity.nihr_open_data import search_nihr_open_data


SOURCE_BLOCKED_TERMS = {
    # Generic source/search words
    "epo",
    "epo ops",
    "lens",
    "lens scholarly",
    "open data",
    "nihr open data",
    "source",
    "sources",
    "anchor",
    "anchors",
    "similarity",
    "similarity check",
    "similarity checker",
    "novelty",
    "test",
    "testing",

    # Generic organisations / schemes / sector labels
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

    # Generic regulatory / readiness terms
    "trl",
    "technology readiness level",
    "technology readiness",
    "readiness level",
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
    "clinical validation",
    "clinical validation needs",
    "clinical safety",
    "quality management system",
    "technical file",
    "technical documentation",
    "post-market surveillance",

    # Generic finance/checklist terms
    "health economics",
    "economic evaluation",
    "economic model",
    "decision-analytic model",
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
    "personal social services",
    "budget",
    "budget impact",
    "cost model",
    "value for money",
    "acord",
    "soecat",

    # Generic document/application terms
    "references",
    "reference list",
    "invention",
    "invention for innovation",
    "i4i",
    "pda",
    "product development award",
    "knowledge",
    "knowledge mobilisation",
    "project management",
    "work package",
    "methodology",
    "aims",
    "objectives",
    "background",
    "rationale",
    "appendix",
    "plain english summary",
    "application",
    "proposal",
    "guidance",
    "funding",
    "grant",

    # Generic evaluation terms
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
}


def _is_public_identifier(term: str) -> bool:
    return is_any_identifier(term)


def _is_patent_identifier(term: str) -> bool:
    return _id_is_patent_identifier(term)


def _is_nihr_identifier(term: str) -> bool:
    return _id_is_nihr_identifier(term)


def _is_trial_identifier(term: str) -> bool:
    return _id_is_trial_identifier(term)


def _looks_like_outcome_scale(term: str) -> bool:
    key = normalise(term)

    if key in {"sus", "pss", "pssuq", "eq-5d", "eq-5d-5l"}:
        return True

    if "qol" in key:
        return True

    if key.endswith(" scale") or key.endswith(" questionnaire") or key.endswith(" inventory"):
        return True

    return False


def _blocked_for_external_query(term: str) -> bool:
    """Return True if a term should not be sent to any external source."""

    if _is_public_identifier(term):
        return False

    key = normalise(term)

    if not key:
        return True

    if key in SOURCE_BLOCKED_TERMS:
        return True

    if _looks_like_outcome_scale(term):
        return True

    if is_generic_term(term):
        return True

    if re.fullmatch(r"trl(?:\s+\d+)?(?:\s+to\s+trl?\s*\d+)?", key):
        return True

    if re.fullmatch(r"(?:ukca|ce)[-\s]?(?:ready|compliant|marked|certified)", key):
        return True

    return False


def _source_term_rank(source: str, term: str) -> tuple[int, str]:
    cls = concept_class(term)

    if source == "EPO OPS":
        if _is_patent_identifier(term):
            return (0, normalise(term))
        if cls == "named_entities":
            return (1, normalise(term))
        if cls == "technical_method_or_mechanism":
            return (2, normalise(term))
        if cls == "product_or_intervention_function":
            return (3, normalise(term))
        if cls == "intervention_type":
            return (4, normalise(term))
        if cls == "clinical_or_social_care_problem":
            return (8, normalise(term))
        return (99, normalise(term))

    if source == "NIHR Open Data":
        if _is_nihr_identifier(term):
            return (0, normalise(term))
        if cls == "named_entities":
            return (1, normalise(term))
        if cls == "intervention_type":
            return (2, normalise(term))
        if cls == "product_or_intervention_function":
            return (3, normalise(term))
        if cls == "technical_method_or_mechanism":
            return (4, normalise(term))
        if cls == "clinical_or_social_care_problem":
            return (5, normalise(term))
        return (99, normalise(term))

    # Lens Scholarly
    if _is_trial_identifier(term):
        return (0, normalise(term))
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


def _ordered_terms_for_source(source: str, primary: list[str], secondary: list[str]) -> list[str]:
    """Route query terms to a source using concept classes, not topic-specific rules."""

    pool = primary + secondary
    out: list[str] = []
    seen: set[str] = set()

    for term in pool:
        key = normalise(term)

        if not key or key in seen:
            continue

        if _blocked_for_external_query(term):
            continue

        cls = concept_class(term)

        if source == "EPO OPS":
            # EPO should receive patent IDs, named interventions/products,
            # technical methods, intervention functions and intervention types.
            # It should not receive NIHR IDs, trial IDs, checklist terms,
            # finance terms or population-only terms.
            if _is_nihr_identifier(term) or _is_trial_identifier(term):
                continue

            allowed = {
                "exact_identifier",
                "named_entities",
                "technical_method_or_mechanism",
                "product_or_intervention_function",
                "intervention_type",
                "clinical_or_social_care_problem",
            }

            if _is_patent_identifier(term) or cls in allowed:
                out.append(term)
                seen.add(key)

        elif source == "NIHR Open Data":
            # NIHR Open Data should receive NIHR IDs first, then named
            # interventions/products, intervention type, function, method and
            # specific clinical/social-care problem.
            # It should not receive patent IDs.
            if _is_patent_identifier(term) or _is_trial_identifier(term):
                continue

            allowed = {
                "exact_identifier",
                "named_entities",
                "intervention_type",
                "product_or_intervention_function",
                "technical_method_or_mechanism",
                "clinical_or_social_care_problem",
            }

            if _is_nihr_identifier(term) or cls in allowed:
                out.append(term)
                seen.add(key)

        else:
            # Lens Scholarly should not receive patent IDs or NIHR award IDs.
            # Trial IDs are useful for scholarly searching.
            if _is_patent_identifier(term) or _is_nihr_identifier(term):
                continue

            allowed = {
                "exact_identifier",
                "named_entities",
                "intervention_type",
                "technical_method_or_mechanism",
                "product_or_intervention_function",
                "clinical_or_social_care_problem",
            }

            if cls in allowed:
                out.append(term)
                seen.add(key)

    return sorted(out, key=lambda item: _source_term_rank(source, item))


def _is_technical_or_function_term(term: str) -> bool:
    return concept_class(term) in {
        "technical_method_or_mechanism",
        "product_or_intervention_function",
        "intervention_type",
    }


def _epo_has_run_basis(terms: list[str]) -> bool:
    """Return True when EPO has enough invention signal to run.

    Valid EPO basis:
    1. patent identifier
    2. named intervention/product plus technical/function/intervention term
    3. at least two technical/function/intervention terms

    EPO should not run on only broad clinical condition, population, setting,
    finance, checklist or document terms.
    """

    if any(_is_patent_identifier(term) for term in terms):
        return True

    named_terms = [
        term for term in terms
        if concept_class(term) == "named_entities"
    ]

    technical_or_function_terms = [
        term for term in terms
        if _is_technical_or_function_term(term)
    ]

    if named_terms and technical_or_function_terms:
        return True

    if len(technical_or_function_terms) >= 2:
        return True

    return False


def _api_terms(source: str, primary: list[str], secondary: list[str]) -> list[str]:
    """Return source-specific safe terms in priority order."""

    ordered = _ordered_terms_for_source(source, primary, secondary)

    if source == "EPO OPS":
        patent_ids = [term for term in ordered if _is_patent_identifier(term)]

        if patent_ids:
            return patent_ids[:3]

        named = [
            term for term in ordered
            if concept_class(term) == "named_entities"
        ]

        technical_or_function = [
            term for term in ordered
            if _is_technical_or_function_term(term)
        ]

        clinical_problem = [
            term for term in ordered
            if concept_class(term) == "clinical_or_social_care_problem"
        ]

        picked: list[str] = []

        # Named product/intervention first, then mechanism/function.
        picked.extend(named[:2])
        picked.extend(technical_or_function[:3])

        # Clinical/social-care problem can support a patent query, but must not
        # be the only reason EPO runs.
        if picked and clinical_problem:
            picked.append(clinical_problem[0])

        return picked[:5]

    if source == "NIHR Open Data":
        nihr_ids = [term for term in ordered if _is_nihr_identifier(term)]

        named = [
            term for term in ordered
            if concept_class(term) == "named_entities"
        ]

        function_or_type_or_method = [
            term for term in ordered
            if concept_class(term) in {
                "intervention_type",
                "product_or_intervention_function",
                "technical_method_or_mechanism",
            }
        ]

        clinical_problem = [
            term for term in ordered
            if concept_class(term) == "clinical_or_social_care_problem"
        ]

        return (nihr_ids + named + function_or_type_or_method + clinical_problem)[:5]

    # Lens Scholarly
    trial_ids = [term for term in ordered if _is_trial_identifier(term)]

    named = [
        term for term in ordered
        if concept_class(term) == "named_entities"
    ]

    technical_or_function = [
        term for term in ordered
        if concept_class(term) in {
            "technical_method_or_mechanism",
            "product_or_intervention_function",
            "intervention_type",
        }
    ]

    clinical_problem = [
        term for term in ordered
        if concept_class(term) == "clinical_or_social_care_problem"
    ]

    return (trial_ids + named + technical_or_function + clinical_problem)[:5]


def _format_query(terms: list[str]) -> str:
    """Build a conservative text query for Lens-style search."""

    cleaned_terms = [term for term in terms if term and not _blocked_for_external_query(term)]

    def quote(term: str) -> str:
        return f'"{term}"' if " " in term else term

    return " OR ".join(quote(term) for term in cleaned_terms[:5])


def _not_run(source: str, message: str, terms: list[str] | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "status": "not_run",
        "matches_found": 0,
        "raw_records_returned": 0,
        "top_match": message,
        "score": 0.0,
        "risk": "NONE",
        "similarity_type": "",
        "matched_concepts": [],
        "specific_matched_concepts": [],
        "generic_matched_concepts": [],
        "matched_dimensions": {},
        "why_relevant": message,
        "link_or_id": "",
        "query_terms_used": terms or [],
    }


def _mock_result(source: str, terms: list[str]) -> dict[str, Any]:
    return {
        "source": source,
        "status": "mock",
        "matches_found": 0,
        "raw_records_returned": 0,
        "top_match": "Mock mode: no live search performed.",
        "score": 0.0,
        "risk": "NONE",
        "similarity_type": "mock_mode",
        "matched_concepts": [],
        "specific_matched_concepts": [],
        "generic_matched_concepts": [],
        "matched_dimensions": {},
        "why_relevant": "Mock mode returns no simulated similarity result.",
        "link_or_id": "",
        "query_terms_used": terms,
    }


def _clean_api_error(exc: Exception) -> str:
    name = exc.__class__.__name__
    return f"API error: {name}; live search failed. Query URL suppressed."


def _join_raw_values(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    chunks: list[str] = []

    for key in keys:
        value = raw.get(key)

        if isinstance(value, list):
            chunks.extend(str(item) for item in value if item)
        elif value:
            chunks.append(str(value))

    return " ".join(chunks)


def _metadata_for_scoring(result: dict[str, Any]) -> tuple[str, str]:
    """Extract title and abstract-like metadata for cautious scoring."""

    title = str(result.get("top_match", "") or "")
    abstract = str(result.get("raw", "") or "")
    raw = result.get("raw")

    if isinstance(raw, dict):
        title = (
            _join_raw_values(
                raw,
                (
                    "title",
                    "project_title",
                    "invention_title",
                    "acronym",
                    "recordid",
                    "project_id",
                ),
            )
            or title
        )

        abstract = (
            _join_raw_values(
                raw,
                (
                    "abstract",
                    "scientific_abstract",
                    "plain_english_abstract",
                    "plain_english_summary",
                    "snippet",
                    "description",
                    "metadata_text",
                    "programme",
                    "funding_stream",
                    "acronym",
                    "project_id",
                    "funding_and_awards_link",
                    "doc_numbers",
                    "publication-number",
                    "applicants",
                    "organisation",
                    "organization",
                ),
            )
            or metadata_text(title, "", raw)
        )

    link_or_id = str(result.get("link_or_id", "") or "")

    if link_or_id and link_or_id not in abstract:
        abstract = f"{abstract} {link_or_id}".strip()

    return title, abstract


def _source_function_pairs() -> list[tuple[Callable[[Any, Settings], dict], str]]:
    return [
        (search_lens, "Lens Scholarly"),
        (search_epo, "EPO OPS"),
        (search_nihr_open_data, "NIHR Open Data"),
    ]


def run_similarity_service(
    facts: ApplicationFacts,
    settings: Settings,
    run_similarity_check: bool = False,
    mock_mode: bool = False,
    snippets: list[str] | None = None,
) -> dict[str, Any]:
    """Run privacy-gated, source-routed similarity checks."""

    query = build_similarity_query(facts, snippets)
    terms = query.primary_terms + query.secondary_terms
    sources = ["Lens Scholarly", "EPO OPS", "NIHR Open Data"]

    if not run_similarity_check:
        return {
            "query": query,
            "results": [
                _not_run(source, "Similarity check disabled by user.", _api_terms(source, query.primary_terms, query.secondary_terms))
                for source in sources
            ],
        }

    meaningful_count = getattr(query, "meaningful_term_count", len(terms))

    if meaningful_count < 2 and not any(_is_public_identifier(term) for term in terms):
        return {
            "query": query,
            "results": [
                _not_run(
                    source,
                    "At least two meaningful application-specific terms are required before live searching.",
                    _api_terms(source, query.primary_terms, query.secondary_terms),
                )
                for source in sources
            ],
        }

    if mock_mode:
        return {
            "query": query,
            "results": [
                _mock_result(source, _api_terms(source, query.primary_terms, query.secondary_terms))
                for source in sources
            ],
        }

    if settings.strict_local_only_mode:
        return {
            "query": query,
            "results": [
                _not_run(
                    source,
                    "Strict local-only mode blocks all external API calls.",
                    _api_terms(source, query.primary_terms, query.secondary_terms),
                )
                for source in sources
            ],
        }

    if not settings.allow_external_similarity_queries:
        return {
            "query": query,
            "results": [
                _not_run(
                    source,
                    "External similarity queries are disabled.",
                    _api_terms(source, query.primary_terms, query.secondary_terms),
                )
                for source in sources
            ],
        }

    if not settings.send_only_safe_query_terms:
        return {
            "query": query,
            "results": [
                _not_run(
                    source,
                    "Safe-query-term privacy gate is disabled.",
                    _api_terms(source, query.primary_terms, query.secondary_terms),
                )
                for source in sources
            ],
        }

    raw_results: list[dict[str, Any]] = []

    for func, source in _source_function_pairs():
        api_terms = _api_terms(source, query.primary_terms, query.secondary_terms)

        if source == "EPO OPS" and not _epo_has_run_basis(api_terms):
            raw_results.append(
                _not_run(
                    source,
                    "EPO OPS requires a patent identifier, a named intervention plus a technical/function term, or at least two technical/function terms after cleaning.",
                    api_terms,
                )
            )
            continue

        if source != "EPO OPS" and len(api_terms) < 2 and not any(_is_public_identifier(term) for term in api_terms):
            raw_results.append(
                _not_run(
                    source,
                    "Fewer than two safe source-specific query terms after cleaning.",
                    api_terms,
                )
            )
            continue

        try:
            request_query: Any
            if source in {"EPO OPS", "NIHR Open Data"}:
                request_query = api_terms
            else:
                request_query = _format_query(api_terms)

            result = func(request_query, settings)

        except Exception as exc:
            result = {
                "source": source,
                "status": "error",
                "matches_found": 0,
                "raw_records_returned": 0,
                "top_match": "",
                "raw": {},
                "score": 0.0,
                "risk": "NONE",
                "similarity_type": "",
                "matched_concepts": [],
                "specific_matched_concepts": [],
                "generic_matched_concepts": [],
                "matched_dimensions": {},
                "why_relevant": _clean_api_error(exc),
                "link_or_id": "",
                "query_terms_used": api_terms,
            }

        raw_records = int(result.get("matches_found", 0) or 0)
        title, abstract = _metadata_for_scoring(result)

        # Score only against source-specific terms. This prevents, for example,
        # EPO-only patent IDs from inflating NIHR Open Data matches and prevents
        # NIHR IDs from inflating patent scoring.
        scored = score_result(
            api_terms,
            title,
            abstract,
            getattr(facts, "acronym_or_short_name", ""),
        )

        result.update(scored)
        result["raw_records_returned"] = raw_records
        result["matches_found"] = 1 if scored["score"] > 0 and scored["risk"] != "NONE" else 0
        result["query_terms_used"] = api_terms

        if result["matches_found"] == 0:
            if source == "NIHR Open Data":
                result["top_match"] = "No relevant NIHR Open Data match found"
            elif source == "EPO OPS":
                result["top_match"] = "No relevant EPO match found"
            else:
                result["top_match"] = "No relevant match found"
        else:
            result["top_match"] = title or result.get("top_match", "")

        raw_results.append(result)

    return {
        "query": query,
        "results": raw_results,
    }