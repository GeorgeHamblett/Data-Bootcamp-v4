"""Cautious, metadata-grounded similarity scoring utilities.

Scoring is domain-agnostic. It must not be hard-coded for a clinical area,
technology type, intervention type or example application.

Core rules:
- exact public identifier overlap = VERY_HIGH
- same named intervention/product plus method/function/problem support = HIGH
- same technical method + function + target problem = HIGH
- technical method + function = MEDIUM
- named product only = MEDIUM at most
- short acronym only = NONE, because acronyms are ambiguous
- clinical/social-care problem only = LOW
- population/setting only = NONE
- generic/checklist/source words only = NONE
"""

from __future__ import annotations

import re
from typing import Any

from similarity.query_builder import concept_class, is_generic_term, normalise
from similarity.identifiers import extract_identifiers, identifier_match_keys


RISK_SCORES = {
    "NONE": 0.0,
    "LOW": 0.15,
    "MEDIUM": 0.45,
    "HIGH": 0.75,
    "VERY_HIGH": 0.95,
    "HUMAN_CHECK": 0.0,
}


GENERIC_OR_SOURCE_TERMS = {
    # Source/search/reference names
    "hra",
    "ukipo",
    "uspto",
    "epo",
    "epo ops",
    "lens",
    "lens scholarly",
    "open data",
    "nihr open data",
    "google patents",
    "patents",
    "patent law",
    "machine-readable",
    "machine readable",
    "source",
    "sources",
    "anchor",
    "anchors",
    "public-source anchor",
    "public-source anchors",
    "reference",
    "references",
    "url",
    "urls",

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
    "cea",
    "cua",
    "qaly",
    "qalys",
    "icer",
    "roi",
    "eq-5d",
    "eq-5d-5l",
    "pss",
    "pssuq",
    "sus",
    "health economics",
    "economic evaluation",
    "budget",
    "budget impact",
    "cost effectiveness",
    "cost-effectiveness",
    "cost utility",
    "cost-utility",
    "outcome",
    "outcomes",
    "endpoint",
    "endpoints",
    "feasibility",
    "acceptability",
    "usability",
    "recruitment",
    "retention",
    "fidelity",
    "interviews",

    # Application/document headings
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
    "application",
    "proposal",
    "guidance",
    "funding",
    "grant",
    "wp1",
    "wp2",
    "wp3",
    "wp4",
    "wp5",
    "wp6",

    # Regulatory/readiness
    "trl",
    "technology readiness level",
    "ukca",
    "ukca-ready",
    "ukca ready",
    "iso 13485",
    "iso 14971",
    "iec 62304",
    "dtac",
    "dtac-aligned",
    "dtac aligned",
    "clinical validation",
    "regulatory readiness",
    "activities-specific",
    "activities specific",
}


INFRASTRUCTURE_CONCEPTS = {
    "isolated execution",
    "execution environment",
    "execution environments",
    "isolation level",
    "isolation levels",
    "resource allocation",
    "computing resource",
    "computing resources",
    "deployment architecture",
    "application isolation",
    "edge artificial intelligence platform",
    "edge ai platform",
}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        key = normalise(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)

    return out


def _is_short_acronym(term: str) -> bool:
    item = str(term or "").strip()
    return bool(re.fullmatch(r"[A-Z0-9]{2,6}", item))


def _is_generic_like(term: str) -> bool:
    key = normalise(term)

    if not key:
        return True

    if key in GENERIC_OR_SOURCE_TERMS:
        return True

    if key.rstrip("s") in GENERIC_OR_SOURCE_TERMS:
        return True

    if is_generic_term(term):
        return True

    if "qol" in key:
        return True

    if key.endswith(" scale") or key.endswith(" questionnaire") or key.endswith(" inventory"):
        return True

    if re.fullmatch(r"(?:ukca|ce)[-\s]?(?:ready|compliant|marked|certified)", key):
        return True

    if re.fullmatch(r"trl(?:\s+\d+)?(?:\s+to\s+trl?\s*\d+)?", key):
        return True

    if re.fullmatch(r"wp\d+", key):
        return True

    if key.endswith(" aligned") or key.endswith(" specific"):
        return True

    if "data collection complete" in key:
        return True

    if "access to sufficiently" in key:
        return True

    return False


def _phrase_in_haystack(haystack: str, phrase: str) -> bool:
    key = normalise(phrase)

    if not key:
        return False

    dehyphenated_haystack = haystack.replace("-", " ")
    dehyphenated_key = key.replace("-", " ")

    if len(key) <= 6 and " " not in key:
        return bool(re.search(rf"\b{re.escape(key)}\b", haystack))

    return (
        key in haystack
        or dehyphenated_key in dehyphenated_haystack
        or key.replace(" ", "-") in haystack
    )


def _identifier_matches(app_terms: list[str], metadata_text: str) -> list[str]:
    app_ids = extract_identifiers(" ".join(app_terms))
    metadata_ids = extract_identifiers(metadata_text)

    if not app_ids or not metadata_ids:
        return []

    app_keys = identifier_match_keys(app_ids)
    metadata_keys = identifier_match_keys(metadata_ids)
    matched_keys = app_keys & metadata_keys

    if not matched_keys:
        return []

    matched_display: list[str] = []

    for item in app_ids:
        if identifier_match_keys([item]) & matched_keys:
            matched_display.append(item)

    return _dedupe(matched_display)


def _exact_identifier_result(exact_ids: list[str]) -> dict[str, Any]:
    return {
        "score": 0.95,
        "risk": "VERY_HIGH",
        "similarity_type": "exact_identifier_match",
        "matched_concepts": exact_ids,
        "specific_matched_concepts": exact_ids,
        "generic_matched_concepts": [],
        "matched_dimensions": {"exact_identifier": exact_ids},
        "why_relevant": (
            "The returned metadata contains the same public award, trial or patent "
            "identifier as the application. This should be treated as a direct "
            "similarity hit requiring manual review."
        ),
    }


def _specific_matches(terms: list[str], haystack: str) -> list[str]:
    matches: list[str] = []

    for term in terms:
        if _is_generic_like(term):
            continue

        cls = concept_class(term)

        if cls in {"generic_document_terms", "population_setting"}:
            continue

        if _phrase_in_haystack(haystack, term):
            matches.append(term)

    return _dedupe(matches)


def matched_concepts(terms: list[str], title: str, abstract: str = "") -> list[str]:
    haystack = normalise(f"{title} {abstract}")
    return _specific_matches(terms, haystack)


def _generic_matches(terms: list[str], haystack: str) -> list[str]:
    matches: list[str] = []

    for term in terms:
        if not _is_generic_like(term):
            continue

        if _phrase_in_haystack(haystack, term):
            matches.append(term)

    return _dedupe(matches)


def _infrastructure_matches(haystack: str) -> list[str]:
    return [
        concept
        for concept in sorted(INFRASTRUCTURE_CONCEPTS)
        if _phrase_in_haystack(haystack, concept)
    ]


def _dimension_for_term(term: str) -> str:
    cls = concept_class(term)

    if cls == "exact_identifier":
        return "exact_identifier"

    if cls == "named_entities":
        return "specific_named_product_or_phrase"

    if cls == "technical_method_or_mechanism":
        return "technical_method"

    if cls == "product_or_intervention_function":
        return "product_function"

    if cls == "intervention_type":
        return "intervention_type"

    if cls == "clinical_or_social_care_problem":
        return "clinical_problem"

    if cls == "population_setting":
        return "target_setting_population"

    return cls


def _matched_dimensions(
    specific: list[str],
    product_or_acronym: str,
    haystack: str,
) -> dict[str, list[str]]:
    dims: dict[str, list[str]] = {}

    for term in specific:
        dim = _dimension_for_term(term)

        if dim in {"generic_document_terms", "population_setting"}:
            continue

        dims.setdefault(dim, []).append(term)

        key = normalise(term)

        if any(
            marker in key
            for marker in (
                "risk",
                "decline",
                "deterioration",
                "fall",
                "falls",
                "ulcer",
                "wound",
                "sepsis",
                "discharge",
                "infection",
                "diagnosis",
                "cancer",
                "stroke",
                "depression",
                "pain",
            )
        ):
            dims.setdefault("clinical_problem", []).append(term)

        if any(
            marker in key
            for marker in (
                "detection",
                "prediction",
                "prevention",
                "rehabilitation",
                "triage",
                "feedback",
                "coordination",
                "planning",
                "support",
                "monitoring",
                "recommendation",
                "classification",
                "stratification",
                "assessment",
            )
        ):
            dims.setdefault("product_function", []).append(term)

    if product_or_acronym and not _is_generic_like(product_or_acronym):
        if _phrase_in_haystack(haystack, product_or_acronym):
            dims.setdefault("specific_named_product_or_phrase", []).append(product_or_acronym)

    return {key: _dedupe(values) for key, values in dims.items()}


def _explanation(
    similarity_type: str,
    risk: str,
    specific: list[str],
    generic: list[str],
    infrastructure: list[str],
) -> str:
    if similarity_type == "no_meaningful_overlap":
        return "No meaningful overlap found in the available metadata."

    if similarity_type == "generic_overlap":
        return (
            "Only generic source, document, checklist, finance or regulatory terms "
            "overlap; these do not indicate external novelty similarity."
        )

    if similarity_type == "acronym_only_ambiguous":
        return (
            "Only a short acronym overlaps. Short acronyms are ambiguous and do "
            "not by themselves indicate that the returned record concerns the same "
            "intervention, method, function or target problem."
        )

    if similarity_type == "infrastructure_only":
        return (
            "The returned record concerns generic infrastructure/platform concepts "
            "without overlap on the application's named intervention, technical "
            "method, product function or target problem."
        )

    if similarity_type == "condition_only_overlap":
        return (
            "Only the broad clinical, public-health or social-care problem overlaps; "
            "this is a low-specificity similarity signal."
        )

    if similarity_type == "specific_name_only":
        return (
            "The returned metadata overlaps on a named product/intervention term, "
            "but lacks supporting overlap on technical method, function or target "
            "problem."
        )

    if similarity_type == "same_method_or_function_only":
        return (
            "There is limited overlap on a method or function, but not enough "
            "metadata overlap on the named intervention and target problem to treat "
            "it as a strong match."
        )

    return (
        f"Potential similarity: overlap spans {len(specific)} specific concept(s), "
        f"including {', '.join(specific[:5])}."
    )


def _only_short_acronym_matches(matches: list[str]) -> bool:
    return bool(matches) and all(_is_short_acronym(match) for match in matches)


def _only_named_product_matches(dimensions: dict[str, list[str]]) -> bool:
    return set(dimensions) == {"specific_named_product_or_phrase"}


def score_result(
    terms: list[str],
    title: str,
    abstract: str = "",
    product_or_acronym: str = "",
) -> dict[str, Any]:
    metadata_text = f"{title} {abstract}"
    haystack = normalise(metadata_text)

    exact_ids = _identifier_matches(terms, metadata_text)

    if exact_ids:
        return _exact_identifier_result(exact_ids)

    matches = _specific_matches(terms, haystack)
    generic = _generic_matches(terms, haystack)
    infrastructure = _infrastructure_matches(haystack)

    if infrastructure and not matches:
        return {
            "score": 0.0,
            "risk": "NONE",
            "similarity_type": "infrastructure_only",
            "matched_concepts": generic + infrastructure,
            "specific_matched_concepts": [],
            "generic_matched_concepts": _dedupe(generic + infrastructure),
            "matched_dimensions": {},
            "why_relevant": _explanation("infrastructure_only", "NONE", [], generic, infrastructure),
        }

    if not matches:
        if generic:
            return {
                "score": 0.0,
                "risk": "NONE",
                "similarity_type": "generic_overlap",
                "matched_concepts": generic,
                "specific_matched_concepts": [],
                "generic_matched_concepts": generic,
                "matched_dimensions": {},
                "why_relevant": _explanation("generic_overlap", "NONE", [], generic, infrastructure),
            }

        return {
            "score": 0.0,
            "risk": "NONE",
            "similarity_type": "no_meaningful_overlap",
            "matched_concepts": [],
            "specific_matched_concepts": [],
            "generic_matched_concepts": [],
            "matched_dimensions": {},
            "why_relevant": _explanation("no_meaningful_overlap", "NONE", [], [], []),
        }

    # Critical false-positive control:
    # a bare short acronym, such as MQAE, CRP or ABC, is not enough to call a
    # returned scholarly/patent/award record relevant.
    if _only_short_acronym_matches(matches):
        return {
            "score": 0.0,
            "risk": "NONE",
            "similarity_type": "acronym_only_ambiguous",
            "matched_concepts": matches,
            "specific_matched_concepts": [],
            "generic_matched_concepts": generic,
            "matched_dimensions": {},
            "why_relevant": _explanation("acronym_only_ambiguous", "NONE", matches, generic, infrastructure),
        }

    dimensions = _matched_dimensions(matches, product_or_acronym, haystack)
    dimension_names = set(dimensions)

    condition_dims = {"clinical_problem", "target_setting_population"}
    invention_dims = {
        "technical_method",
        "product_function",
        "intervention_type",
        "specific_named_product_or_phrase",
    }

    if not dimensions:
        risk = "NONE"
        similarity_type = "no_meaningful_overlap"

    elif _only_named_product_matches(dimensions):
        risk = "MEDIUM"
        similarity_type = "specific_name_only"

    elif dimension_names <= condition_dims:
        risk = "LOW"
        similarity_type = "condition_only_overlap"

    elif "specific_named_product_or_phrase" in dimension_names and (
        dimension_names & {"technical_method", "product_function", "intervention_type", "clinical_problem"}
    ):
        risk = "HIGH"
        similarity_type = "direct_match"

    elif {"technical_method", "product_function", "clinical_problem"} <= dimension_names:
        risk = "HIGH"
        similarity_type = "direct_match"

    elif {"technical_method", "product_function"} <= dimension_names:
        risk = "MEDIUM"
        similarity_type = "same_method_and_function"

    elif dimension_names & invention_dims and dimension_names & condition_dims:
        risk = "MEDIUM"
        similarity_type = "same_domain_broad"

    elif dimension_names & invention_dims:
        risk = "LOW"
        similarity_type = "same_method_or_function_only"

    else:
        risk = "LOW"
        similarity_type = "condition_only_overlap"

    score = min(
        RISK_SCORES[risk],
        0.15 + 0.18 * len(matches) + 0.08 * len(dimension_names),
    )

    if similarity_type == "condition_only_overlap":
        score = min(score, 0.20)

    if similarity_type == "specific_name_only":
        score = min(score, 0.45)

    if risk == "HIGH":
        score = max(score, 0.70)

    return {
        "score": round(score, 2),
        "risk": risk,
        "similarity_type": similarity_type,
        "matched_concepts": _dedupe(matches + generic),
        "specific_matched_concepts": matches,
        "generic_matched_concepts": generic,
        "matched_dimensions": dimensions,
        "why_relevant": _explanation(similarity_type, risk, matches, generic, infrastructure),
    }


def score_profiles(
    app_profile,
    metadata_profile,
    app_text: str = "",
    metadata_raw_text: str = "",
) -> dict[str, Any]:
    terms: list[str] = []

    for values in app_profile.specific_groups().values():
        terms.extend(values)

    metadata_terms: list[str] = []

    for values in metadata_profile.specific_groups().values():
        metadata_terms.extend(values)

    return score_result(
        _dedupe(terms),
        " ".join(metadata_terms),
        metadata_raw_text,
    )