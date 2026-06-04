"""Domain-agnostic concept extraction for external similarity scoring.

This module builds ConceptProfile objects for applications and returned metadata.
It must not be hard-coded for a clinical area, technology type, funding stream or
example application.

The aim is to classify short, safe phrases into reusable concept dimensions:
- exact identifiers
- named interventions/products/acronyms
- intervention type
- technical method/mechanism
- product or intervention function
- clinical/public-health/social-care problem
- population/setting as weak context
- generic document/checklist/source terms to ignore
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from schemas import ApplicationFacts, NOT_EXPLICITLY_STATED
from similarity.identifiers import extract_identifiers, is_any_identifier
from similarity.query_builder import concept_class, is_generic_term, normalise


TECH_SUFFIXES = (
    "algorithm",
    "model",
    "classifier",
    "engine",
    "sensor",
    "sensors",
    "assay",
    "test",
    "imaging",
    "imager",
    "device",
    "platform",
    "dashboard",
    "tool",
    "pathway",
    "programme",
    "program",
    "package",
    "intervention",
    "biomarker",
    "biomarkers",
    "risk score",
    "risk model",
    "decision support",
    "remote monitoring",
    "assessment",
    "prediction",
    "detection",
    "monitoring",
    "diagnosis",
    "triage",
    "classification",
    "stratification",
    "feedback",
    "coaching",
    "rehabilitation",
    "software",
    "therapeutic",
    "diagnostic",
)

FUNCTION_SUFFIXES = (
    "prevention",
    "rehabilitation",
    "detection",
    "prediction",
    "monitoring",
    "diagnosis",
    "triage",
    "prioritisation",
    "prioritization",
    "risk scoring",
    "risk prediction",
    "decision support",
    "feedback",
    "coaching",
    "escalation",
    "care coordination",
    "care planning",
    "adherence support",
    "symptom management",
    "medication optimisation",
    "medication optimization",
    "referral support",
    "transitional support",
    "assessment",
    "measurement",
)

INTERVENTION_TYPE_SUFFIXES = (
    "therapeutic",
    "programme",
    "program",
    "platform",
    "device",
    "test",
    "assay",
    "pathway",
    "package",
    "tool",
    "intervention",
    "dashboard",
    "app",
    "software",
)

PROBLEM_HINTS = (
    "risk",
    "decline",
    "deterioration",
    "prevention",
    "rehabilitation",
    "disease",
    "condition",
    "syndrome",
    "wound",
    "wounds",
    "ulcer",
    "ulcers",
    "falls",
    "fall",
    "balance",
    "mobility",
    "pain",
    "infection",
    "cancer",
    "diabetes",
    "stroke",
    "frailty",
    "depression",
    "diagnosis",
    "inequality",
    "burden",
    "discharge",
    "sepsis",
    "triage",
    "relapse",
    "delay",
    "deterioration",
)

SETTING_HINTS = (
    "community",
    "clinic",
    "clinics",
    "hospital",
    "primary care",
    "emergency department",
    "care home",
    "care homes",
    "social care",
    "rehabilitation service",
    "rehabilitation services",
    "community service",
    "community services",
    "nursing team",
    "nursing teams",
)

GENERIC_METHOD_PATTERNS = (
    r"(?:deep\s+)?(?:convolutional\s+)?neural\s+network(?:\s+classifier)?",
    r"machine\s+learning(?:\s+model)?",
    r"artificial\s+intelligence(?:\s+model)?",
    r"sequence\s+modelling",
    r"temporal\s+sequence\s+modelling",
    r"hidden\s+markov\s+models?",
    r"random\s+forest",
    r"transformer\s+model",
    r"motion\s+analysis(?:\s+systems?)?",
    r"inertial\s+measurement\s+unit",
    r"accelerometers?",
    r"gyroscopes?",
    r"biomarker\s+panel",
    r"multiplex\s+biomarker\s+panel",
    r"point-of-care\s+test",
    r"self-management\s+coaching",
    r"care\s+pathway\s+redesign",
    r"behaviour-change\s+programme",
    r"behavior-change\s+program",
    r"remote\s+monitoring\s+system",
    r"risk\s+prediction\s+model",
    r"decision[-\s]support\s+(?:tool|system|platform)",
    r"(?:thermal|multispectral|ultrasound|mri|ct)\s+imaging",
    r"(?:pcr|elisa|mass\s+spectrometry)",
    r"(?:shap|lime)",
)

INFRASTRUCTURE_TERMS = {
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

UNRELATED_DOMAIN_SIGNALS = {
    "drug_biologic": {
        "inhibitor",
        "inhibitors",
        "compound",
        "compounds",
        "therapeutic compound",
        "dose",
        "pharmaceutical",
        "small molecule",
        "agonist",
        "antagonist",
        "enzyme",
        "receptor",
    },
    "gene_cell_animal": {
        "gene",
        "genes",
        "protein",
        "mutation",
        "cell",
        "cells",
        "transgenic",
        "mouse",
        "mice",
        "animal model",
        "prion",
        "cjd",
        "senescence",
        "metabolic defect",
    },
    "computing_infrastructure": INFRASTRUCTURE_TERMS,
    "unrelated_engineering": {
        "valve",
        "actuator",
        "circuit",
        "semiconductor",
        "gearbox",
        "battery terminal",
    },
}


@dataclass
class ConceptProfile:
    exact_identifiers: list[str] = field(default_factory=list)
    named_entities: list[str] = field(default_factory=list)
    intervention_type_terms: list[str] = field(default_factory=list)
    technical_method_or_mechanism_terms: list[str] = field(default_factory=list)
    technical_method_terms: list[str] = field(default_factory=list)
    clinical_condition_terms: list[str] = field(default_factory=list)
    clinical_problem_terms: list[str] = field(default_factory=list)
    product_function_terms: list[str] = field(default_factory=list)
    population_setting_terms: list[str] = field(default_factory=list)
    generic_domain_terms: list[str] = field(default_factory=list)
    generic_document_terms: list[str] = field(default_factory=list)
    infrastructure_terms: list[str] = field(default_factory=list)
    domain_signals: dict[str, list[str]] = field(default_factory=dict)
    metadata_incomplete: bool = False

    def specific_groups(self) -> dict[str, list[str]]:
        return {
            "exact_identifier": self.exact_identifiers,
            "named_entity": self.named_entities,
            "intervention_type": self.intervention_type_terms,
            "technical_method": self.technical_method_or_mechanism_terms or self.technical_method_terms,
            "clinical_condition": self.clinical_condition_terms or self.clinical_problem_terms,
            "product_function": self.product_function_terms,
            "population_setting": self.population_setting_terms,
        }


def _clean(value: str) -> str:
    text = str(value or "")
    text = re.sub(
        r"FOR\s+TRAINING\s+USE\s+ONLY|FICTIONAL\s+EXAMPLE\s+APPLICATION|SYNTHETIC\s+EXEMPLAR|DUMMY\s+APPLICATION",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+", " ", text).strip(" .;:,\n\t")
    return text.strip(" ,;:-/")


def _looks_like_outcome_scale(term: str) -> bool:
    key = normalise(term)

    if key in {"sus", "pss", "pssuq", "eq-5d", "eq-5d-5l"}:
        return True

    if "qol" in key:
        return True

    if key.endswith(" scale") or key.endswith(" questionnaire") or key.endswith(" inventory"):
        return True

    return False


def _looks_like_regulatory_or_workpackage(term: str) -> bool:
    key = normalise(term)

    if re.fullmatch(r"wp\d+", key):
        return True

    if key.endswith(" aligned") or key.endswith("-aligned"):
        return True

    if key.endswith(" ready") or key.endswith("-ready"):
        return True

    if key.endswith(" specific") or key.endswith("-specific"):
        return True

    if re.fullmatch(r"trl(?:\s+\d+)?(?:\s+to\s+trl?\s*\d+)?", key):
        return True

    if re.fullmatch(r"(?:ukca|ce)[-\s]?(?:ready|compliant|marked|certified)", key):
        return True

    return False


def _looks_like_sentence_fragment(term: str) -> bool:
    key = normalise(term)
    words = key.split()

    if is_any_identifier(term):
        return False

    bad_patterns = (
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
    )

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


def _valid_concept(term: str) -> bool:
    value = _clean(term)
    key = normalise(value)

    if not value or not key or value == NOT_EXPLICITLY_STATED:
        return False

    if is_any_identifier(value):
        return True

    if is_generic_term(value):
        return False

    if _looks_like_outcome_scale(value):
        return False

    if _looks_like_regulatory_or_workpackage(value):
        return False

    if _looks_like_sentence_fragment(value):
        return False

    if len(value) > 90:
        return False

    if len(key.split()) > 8:
        return False

    return True


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        value = _clean(str(item))
        key = normalise(value)

        if not key or key in seen:
            continue

        if not _valid_concept(value):
            continue

        seen.add(key)
        out.append(value)

    return out


def _dedupe_all(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        value = _clean(str(item))
        key = normalise(value)

        if key and key not in seen:
            seen.add(key)
            out.append(value)

    return out


def _text_from_facts(facts: ApplicationFacts, fields: list[str]) -> str:
    chunks: list[str] = []

    for field_name in fields:
        value = getattr(facts, field_name, "")

        if isinstance(value, list):
            chunks.extend(str(item) for item in value if str(item).strip())
        elif value and value != NOT_EXPLICITLY_STATED:
            chunks.append(str(value))

    return ". ".join(chunks)


def _identifier_terms(text: str) -> list[str]:
    return extract_identifiers(text)


def _generic_method_terms(text: str) -> list[str]:
    phrases: list[str] = []

    for pattern in GENERIC_METHOD_PATTERNS:
        for match in re.finditer(rf"\b{pattern}\b", text, re.I):
            phrases.append(match.group(0).strip(" .;:,/-"))

    return _dedupe(phrases)


def _suffix_phrases(text: str, suffixes: tuple[str, ...], max_words: int = 8) -> list[str]:
    phrases: list[str] = []
    suffix_re = "|".join(re.escape(suffix) for suffix in sorted(suffixes, key=len, reverse=True))

    for match in re.finditer(
        rf"\b(?:[A-Za-z0-9+#-]+\s+){{1,{max_words - 1}}}(?:{suffix_re})\b",
        text,
        re.I,
    ):
        phrase = match.group(0).strip(" .;:,/-")
        words = phrase.split()

        if 2 <= len(words) <= max_words:
            phrases.append(phrase)

            for window in (3, 4, 5, 6):
                if len(words) >= window:
                    phrases.append(" ".join(words[-window:]))

    return _dedupe(phrases)


def _capitalised_entities(text: str) -> list[str]:
    entities: list[str] = []

    # Mixed-case product-like names, hyphenated names and non-generic acronyms.
    token_pattern = (
        r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"
        r"|\b[A-Z][A-Z0-9]{2,14}\b"
        r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"
    )

    for match in re.finditer(token_pattern, text):
        phrase = match.group(0).strip(" .;:,/-")

        if _valid_concept(phrase):
            entities.append(phrase)

    # Multi-word title-case technical/product names only if method-like.
    for match in re.finditer(
        r"\b[A-Z][a-z][A-Za-z0-9]{2,20}(?:\s+[A-Z][a-z][A-Za-z0-9]{2,20}){1,6}\b",
        text,
    ):
        phrase = match.group(0).strip(" .;:,/-")
        key = normalise(phrase)

        if any(keyword in key for keyword in TECH_SUFFIXES + INTERVENTION_TYPE_SUFFIXES):
            entities.append(phrase)

    return _dedupe(entities)


def _clinical_problem_terms(text: str) -> list[str]:
    phrases: list[str] = []
    hints = "|".join(re.escape(hint) for hint in PROBLEM_HINTS)

    for match in re.finditer(
        rf"\b(?:[A-Za-z0-9+#-]+\s+){{0,4}}(?:{hints})(?:\s+[A-Za-z0-9+#-]+){{0,3}}\b",
        text,
        re.I,
    ):
        phrase = match.group(0).strip(" .;:,/-")
        words = phrase.split()

        if 2 <= len(words) <= 6:
            phrases.append(phrase)

    return _dedupe(phrases)


def _setting_terms(text: str) -> list[str]:
    phrases: list[str] = []

    for hint in SETTING_HINTS:
        for match in re.finditer(
            rf"\b(?:[A-Za-z0-9+#-]+\s+){{0,4}}{re.escape(hint)}(?:\s+[A-Za-z0-9+#-]+){{0,3}}\b",
            text,
            re.I,
        ):
            phrase = match.group(0).strip(" .;:,/-")
            words = phrase.split()

            if 2 <= len(words) <= 7:
                phrases.append(phrase)

    return _dedupe(phrases)


def _generic_terms(text: str) -> list[str]:
    terms: list[str] = []

    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9+#/-]*(?:\s+[A-Za-z][A-Za-z0-9+#/-]*){0,4}\b", text):
        phrase = match.group(0).strip(" .;:,/-")

        if is_generic_term(phrase) or _looks_like_outcome_scale(phrase) or _looks_like_regulatory_or_workpackage(phrase):
            terms.append(phrase)

    return _dedupe_all(terms)


def _infra_terms(text: str) -> list[str]:
    key_text = normalise(text)

    return _dedupe_all(
        term
        for term in INFRASTRUCTURE_TERMS
        if re.search(r"\b" + re.escape(normalise(term)) + r"\b", key_text)
    )


def _domain_signals(text: str) -> dict[str, list[str]]:
    key_text = normalise(text)
    found: dict[str, list[str]] = {}

    for category, terms in UNRELATED_DOMAIN_SIGNALS.items():
        hits = _dedupe_all(
            term
            for term in terms
            if re.search(r"\b" + re.escape(normalise(term)) + r"\b", key_text)
        )

        if hits:
            found[category] = hits

    return found


def _split_by_class(terms: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {
        "exact_identifier": [],
        "named_entities": [],
        "intervention_type": [],
        "technical_method_or_mechanism": [],
        "product_or_intervention_function": [],
        "clinical_or_social_care_problem": [],
        "population_setting": [],
        "generic_document_terms": [],
    }

    for term in terms:
        if not _valid_concept(term) and not is_generic_term(term):
            continue

        cls = concept_class(term)

        grouped.setdefault(cls, []).append(term)

    return {key: _dedupe_all(values) for key, values in grouped.items()}


def _all_candidate_terms(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_identifier_terms(text))
    candidates.extend(_capitalised_entities(text))
    candidates.extend(_generic_method_terms(text))
    candidates.extend(_suffix_phrases(text, TECH_SUFFIXES))
    candidates.extend(_suffix_phrases(text, FUNCTION_SUFFIXES))
    candidates.extend(_suffix_phrases(text, INTERVENTION_TYPE_SUFFIXES))
    candidates.extend(_clinical_problem_terms(text))
    candidates.extend(_setting_terms(text))
    return _dedupe(candidates)


def extract_similarity_concepts(facts: ApplicationFacts) -> ConceptProfile:
    named_text = _text_from_facts(
        facts,
        [
            "project_title",
            "product_or_intervention",
            "acronym_or_short_name",
        ],
    )

    technical_text = _text_from_facts(
        facts,
        [
            "technology_type",
            "product_or_intervention",
            "methodology",
            "study_design",
            "regulatory_plan",
            "mechanism_of_action",
        ],
    )

    clinical_text = _text_from_facts(
        facts,
        [
            "clinical_or_social_care_need",
            "target_population",
            "endpoints",
            "market_or_impact_evidence",
        ],
    )

    function_text = _text_from_facts(
        facts,
        [
            "product_or_intervention",
            "clinical_or_social_care_need",
            "endpoints",
            "study_design",
            "methodology",
        ],
    )

    setting_text = _text_from_facts(
        facts,
        [
            "target_population",
            "sites_or_setting",
        ],
    )

    all_text = " ".join([named_text, technical_text, clinical_text, function_text, setting_text])

    named_terms = _dedupe(
        _identifier_terms(named_text)
        + _capitalised_entities(named_text)
        + _suffix_phrases(named_text, INTERVENTION_TYPE_SUFFIXES)
    )

    technical_terms = _dedupe(
        _generic_method_terms(technical_text)
        + _suffix_phrases(technical_text, TECH_SUFFIXES)
    )

    intervention_type_terms = _dedupe(
        _suffix_phrases(technical_text, INTERVENTION_TYPE_SUFFIXES)
    )

    function_terms = _dedupe(
        _suffix_phrases(function_text, FUNCTION_SUFFIXES)
    )

    clinical_terms = _clinical_problem_terms(clinical_text)
    setting_terms = _setting_terms(setting_text)

    grouped = _split_by_class(
        named_terms
        + technical_terms
        + intervention_type_terms
        + function_terms
        + clinical_terms
        + setting_terms
        + _identifier_terms(all_text)
    )

    return ConceptProfile(
        exact_identifiers=_dedupe_all(_identifier_terms(all_text)),
        named_entities=_dedupe_all(grouped.get("named_entities", [])),
        intervention_type_terms=_dedupe_all(
            grouped.get("intervention_type", []) + intervention_type_terms
        ),
        technical_method_or_mechanism_terms=_dedupe_all(
            grouped.get("technical_method_or_mechanism", []) + technical_terms
        ),
        technical_method_terms=_dedupe_all(
            grouped.get("technical_method_or_mechanism", []) + technical_terms
        ),
        clinical_condition_terms=_dedupe_all(
            grouped.get("clinical_or_social_care_problem", []) + clinical_terms
        ),
        clinical_problem_terms=_dedupe_all(
            grouped.get("clinical_or_social_care_problem", []) + clinical_terms
        ),
        product_function_terms=_dedupe_all(
            grouped.get("product_or_intervention_function", []) + function_terms
        ),
        population_setting_terms=_dedupe_all(
            grouped.get("population_setting", []) + setting_terms
        ),
        generic_domain_terms=_generic_terms(all_text),
        generic_document_terms=_generic_terms(all_text),
        infrastructure_terms=_infra_terms(all_text),
        domain_signals=_domain_signals(all_text),
    )


def metadata_text(title: str = "", abstract: str = "", raw: Any = None) -> str:
    chunks = [str(title or ""), str(abstract or "")]

    if isinstance(raw, dict):
        keys = (
            "title",
            "project_title",
            "invention_title",
            "abstract",
            "scientific_abstract",
            "plain_english_abstract",
            "plain_english_summary",
            "snippet",
            "description",
            "metadata_text",
            "acronym",
            "project_id",
            "recordid",
            "funding_and_awards_link",
            "applicants",
            "organisation",
            "organization",
            "doc_numbers",
            "publication-number",
            "publication_number",
            "country",
            "kind",
        )

        for key in keys:
            value = raw.get(key)

            if isinstance(value, list):
                chunks.extend(str(item) for item in value if item)
            elif value:
                chunks.append(str(value))

    elif raw:
        chunks.append(str(raw))

    return " ".join(chunks)


def extract_metadata_concepts(
    title: str = "",
    abstract: str = "",
    raw: Any = None,
) -> ConceptProfile:
    raw_text = metadata_text(title, abstract, raw)
    title_abstract_missing = not normalise(f"{title} {abstract}")

    candidates = _all_candidate_terms(raw_text)
    grouped = _split_by_class(candidates)

    return ConceptProfile(
        exact_identifiers=_dedupe_all(_identifier_terms(raw_text)),
        named_entities=_dedupe_all(grouped.get("named_entities", [])),
        intervention_type_terms=_dedupe_all(grouped.get("intervention_type", [])),
        technical_method_or_mechanism_terms=_dedupe_all(
            grouped.get("technical_method_or_mechanism", [])
        ),
        technical_method_terms=_dedupe_all(
            grouped.get("technical_method_or_mechanism", [])
        ),
        clinical_condition_terms=_dedupe_all(
            grouped.get("clinical_or_social_care_problem", [])
        ),
        clinical_problem_terms=_dedupe_all(
            grouped.get("clinical_or_social_care_problem", [])
        ),
        product_function_terms=_dedupe_all(
            grouped.get("product_or_intervention_function", [])
        ),
        population_setting_terms=_dedupe_all(grouped.get("population_setting", [])),
        generic_domain_terms=_generic_terms(raw_text),
        generic_document_terms=_generic_terms(raw_text),
        infrastructure_terms=_infra_terms(raw_text),
        domain_signals=_domain_signals(raw_text),
        metadata_incomplete=title_abstract_missing and bool(normalise(str(raw or ""))),
    )