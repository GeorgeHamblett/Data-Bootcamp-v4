"""Domain-agnostic similarity checking for NIHR/RSS application review.

This package extracts privacy-preserving similarity concepts from uploaded
funding applications and routes short, safe terms to configured external
sources such as Lens, EPO OPS and NIHR Open Data.

The similarity layer should stay domain-agnostic. It must not be hard-coded
for one clinical area, technology type, programme, intervention class or
example application. Query extraction should prioritise:

1. exact public identifiers
2. named interventions, products, studies and acronyms
3. technical methods, mechanisms or active components
4. product or intervention functions
5. specific clinical, public health or social care problems
6. population and setting terms as weak supporting context only

Application headings, checklist terms, finance terms, generic research terms
and sentence fragments should not be used as external similarity concepts.
"""

from similarity.identifiers import (
    extract_identifiers,
    is_any_identifier,
    is_nihr_identifier,
    is_patent_identifier,
    is_trial_identifier,
    normalise_identifier,
)

__all__ = [
    "extract_identifiers",
    "is_any_identifier",
    "is_nihr_identifier",
    "is_patent_identifier",
    "is_trial_identifier",
    "normalise_identifier",
]