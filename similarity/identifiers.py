"""Shared public identifier extraction and routing helpers for similarity checks.

These helpers are deliberately domain-agnostic. They identify public IDs that are
safe to use as external similarity-search terms, such as patent publication IDs,
NIHR award/project IDs, trial IDs and DOIs.

They do not extract or return full application text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


# ---------------------------------------------------------------------------
# Public identifier patterns
# ---------------------------------------------------------------------------

# Allows compact and spaced patent formats:
# - US20210201479A1
# - US 20210201479 A1
# - US2021201479A1
# - EP4678231A1
# - EP 4 678 231 A1
# - WO2020123456A1
PATENT_RE = re.compile(
    r"\b(?:US|EP|WO)\s*[\d\s]{6,}\s*(?:[A-Z]\d?)?\b",
    re.I,
)

NIHR_RE = re.compile(
    r"\bAI[_\-\s]?AWARD\d{3,}\b|\bNIHR\d{4,}\b",
    re.I,
)

TRIAL_RE = re.compile(
    r"\bISRCTN\d{6,}\b|\bNCT\d{8}\b",
    re.I,
)

DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.I,
)

# Common patent kind codes. Used only for comparison/search keys.
PATENT_KIND_RE = re.compile(r"(A\d|B\d|C\d|U\d|Y\d)$", re.I)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IdentifierGroups:
    """Grouped identifiers extracted from local application text."""

    patent_ids: list[str] = field(default_factory=list)
    nihr_ids: list[str] = field(default_factory=list)
    trial_ids: list[str] = field(default_factory=list)
    dois: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        return _dedupe(
            self.nihr_ids
            + self.patent_ids
            + self.trial_ids
            + self.dois
        )


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        key = str(item or "").upper()

        if item and key not in seen:
            seen.add(key)
            out.append(item)

    return out


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def normalise_identifier(value: str) -> str:
    """Normalise an identifier for display and routing.

    Examples:
    - "US 20210201479 A1" -> "US20210201479A1"
    - "US 2021 0201479 A1" -> "US20210201479A1"
    - "EP 4 678 231 A1" -> "EP4678231A1"
    - "AI-AWARD01723" -> "AI_AWARD01723"
    - "ai award01723" -> "AI_AWARD01723"
    """

    item = str(value or "").strip().strip(".,;:()[]{}<>")
    item = _compact(item)

    # Standardise AI Award separator before matching.
    item = re.sub(r"AI[_\-\s]?AWARD", "AI_AWARD", item, flags=re.I)

    if PATENT_RE.fullmatch(item):
        return item.upper()

    if NIHR_RE.fullmatch(item):
        return item.upper()

    if TRIAL_RE.fullmatch(item):
        return item.upper()

    if DOI_RE.fullmatch(item):
        return item.strip().rstrip(".,;").lower()

    return item


def patent_match_key(value: str) -> str:
    """Return the main patent comparison/search key without the kind code.

    Examples:
    - US20210201479A1 -> US20210201479
    - US11599998B2 -> US11599998
    - EP4678231A1 -> EP4678231
    """

    item = normalise_identifier(value)

    if not PATENT_RE.fullmatch(item):
        return item.upper()

    return PATENT_KIND_RE.sub("", item).upper()


def patent_match_keys(value: str) -> set[str]:
    """Return flexible comparison keys for patent identifiers.

    This handles source-format differences, especially US publication numbers.

    Examples:
    - US20210201479A1 gives:
      - US20210201479
      - US2021201479

    - US2021201479A1 gives:
      - US2021201479
      - US20210201479

    - US11599998B2 gives:
      - US11599998
    """

    item = normalise_identifier(value)

    if not PATENT_RE.fullmatch(item):
        return {item.upper()} if item else set()

    base = patent_match_key(item)
    keys = {base}

    # US publication numbers often appear in two forms:
    # Public/patents form: US + YYYY + 7-digit serial, often with a leading 0
    # EPO-style form:      US + YYYY + serial without that leading 0
    #
    # Example:
    # US20210201479A1 -> US20210201479
    # EPO may expose:   US2021201479A1
    match = re.fullmatch(r"US(\d{4})(\d+)", base)

    if match:
        year, serial = match.groups()

        if len(serial) == 7 and serial.startswith("0"):
            keys.add(f"US{year}{serial[1:]}")

        if len(serial) == 6:
            keys.add(f"US{year}0{serial}")

    return keys


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

def is_patent_identifier(value: str) -> bool:
    return bool(PATENT_RE.fullmatch(normalise_identifier(value)))


def is_nihr_identifier(value: str) -> bool:
    return bool(NIHR_RE.fullmatch(normalise_identifier(value)))


def is_trial_identifier(value: str) -> bool:
    return bool(TRIAL_RE.fullmatch(normalise_identifier(value)))


def is_doi_identifier(value: str) -> bool:
    return bool(DOI_RE.fullmatch(normalise_identifier(value)))


def is_any_identifier(value: str) -> bool:
    item = normalise_identifier(value)

    return (
        is_patent_identifier(item)
        or is_nihr_identifier(item)
        or is_trial_identifier(item)
        or is_doi_identifier(item)
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_identifier_groups(text: str) -> IdentifierGroups:
    """Extract grouped public identifiers from local text."""

    value = str(text or "")

    patent_ids = [
        normalise_identifier(match.group(0))
        for match in PATENT_RE.finditer(value)
    ]

    nihr_ids = [
        normalise_identifier(match.group(0))
        for match in NIHR_RE.finditer(value)
    ]

    trial_ids = [
        normalise_identifier(match.group(0))
        for match in TRIAL_RE.finditer(value)
    ]

    dois = [
        normalise_identifier(match.group(0))
        for match in DOI_RE.finditer(value)
    ]

    return IdentifierGroups(
        patent_ids=_dedupe(patent_ids),
        nihr_ids=_dedupe(nihr_ids),
        trial_ids=_dedupe(trial_ids),
        dois=_dedupe(dois),
    )


def extract_identifiers(text: str) -> list[str]:
    """Extract all public identifiers in source-routing priority order.

    Priority:
    1. NIHR IDs
    2. patent IDs
    3. trial IDs
    4. DOIs
    """

    return extract_identifier_groups(text).all()


def identifier_match_keys(values: list[str]) -> set[str]:
    """Return comparison keys for exact-ID matching.

    Patent IDs are compared using flexible patent keys. This allows
    US20210201479A1 to match US2021201479A1 when EPO/OPS removes the publication
    serial leading zero.

    Other IDs are compared using their normalised display form.
    """

    keys: set[str] = set()

    for value in values:
        item = normalise_identifier(value)

        if not item:
            continue

        if is_patent_identifier(item):
            keys.update(patent_match_keys(item))
        else:
            keys.add(item.upper())

    return keys