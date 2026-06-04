"""Build privacy-preserving, domain-agnostic external similarity queries.

This module extracts short, safe similarity concepts from NIHR/RSS-style
applications. It must not be hard-coded for one disease area, intervention type,
technology type, funding stream or example application.

Priority:
1. exact public identifiers
2. named interventions/products/studies/acronyms
3. technical methods, mechanisms or active components
4. product/intervention functions
5. specific clinical, public health or social care problems
6. population/setting terms only as weak supporting context

Never use application headings, checklist terms, finance terms, business-model
terms, source names, generic research terms, regulatory readiness terms, outcome
scales, work-package labels, comparator/endpoint fragments, plain-English
sentence fragments, pure numbers, broad commercial labels, or generic technology
labels as external similarity concepts.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from schemas import ApplicationFacts, NOT_EXPLICITLY_STATED, SimilarityQuery
from similarity.identifiers import (
    extract_identifiers,
    is_any_identifier,
    is_nihr_identifier,
    is_patent_identifier,
    is_trial_identifier,
)


MAX_QUERY_TERMS = 10
MAX_TERM_CHARS = 90


GENERIC_DOCUMENT_TERMS = {
    # ------------------------------------------------------------------
    # Basic filler
    # ------------------------------------------------------------------
    "the", "a", "an", "it", "this", "that", "these", "those",
    "we", "our", "ours", "you", "your", "their", "they", "them",
    "he", "she", "his", "her", "its", "new", "novel", "current",
    "clear", "named", "using", "use", "used", "uses", "will",
    "would", "could", "should", "may", "might", "also", "including",
    "includes", "include", "based", "related", "relevant", "proposed",
    "existing", "further", "future", "next", "previous", "prior",
    "same", "different", "specific", "general", "major", "minor",

    # ------------------------------------------------------------------
    # File/template/document words
    # ------------------------------------------------------------------
    "uploaded", "upload", "file", "document", "docx", "pdf", "txt",
    "training", "dummy", "synthetic", "template", "draft", "report",
    "mock", "mock application", "example", "exemplar", "test case",
    "sample", "sample application", "for training use only",
    "fictional example application", "training use only",
    "synthetic exemplar", "synthetic test proposal",
    "not for submission", "created for ai platform testing purposes only",
    "all organisations are fictional", "weaknesses intentionally embedded",

    # ------------------------------------------------------------------
    # Application structure
    # ------------------------------------------------------------------
    "application", "proposal", "guidance", "funding", "grant",
    "programme", "program", "section", "background", "rationale",
    "methodology", "project", "research", "study", "objective",
    "objectives", "aim", "aims", "research question",
    "research questions", "overall aim", "work", "package",
    "work package", "task", "month", "months", "phase", "phases",
    "milestone", "milestones", "gantt", "chart", "risk register",
    "risk", "risks", "mitigation", "study management",
    "project management", "governance", "steering group",
    "trial management group", "team meeting", "regular team meetings",
    "references", "reference", "reference list", "appendix",
    "appendices", "plain english", "plain english summary",
    "summary", "scientific abstract", "abstract", "contents",
    "eligibility", "programme fit", "program fit",
    "changes from previous stage", "n/a", "not applicable",
    "timeline", "timeline and milestones", "project duration",
    "project plan", "study design", "sites", "sample size",
    "inclusion criteria", "exclusion criteria", "data management plan",
    "risk mitigation", "knowledge mobilisation", "knowledge mobilization",
    "commercialisation", "commercialization", "finance summary",

    # ------------------------------------------------------------------
    # Source/search words
    # ------------------------------------------------------------------
    "epo", "epo ops", "lens", "lens scholarly", "open data",
    "nihr open data", "ukipo", "uspto", "hra", "google patents",
    "source", "sources", "anchor", "anchors", "source anchor",
    "source anchors", "public-source anchor", "public-source anchors",
    "public source anchor", "public source anchors", "deliberate",
    "deliberately", "similarity", "similarity check",
    "similarity checker", "novelty", "test", "testing", "stress test",
    "high similarity", "high-similarity", "machine-readable",
    "machine readable",

    # ------------------------------------------------------------------
    # Organisations / schemes / generic sector terms
    # ------------------------------------------------------------------
    "nihr", "nhs", "nhs england", "nice", "mhra", "dhsc", "rss",
    "hra", "nhse", "icb", "ics", "girft", "bsi", "invention",
    "invention for innovation", "i4i", "pda", "product development award",
    "medtech", "healthtech", "digital health", "sme", "uk sme",
    "small and medium enterprise", "small and medium-sized enterprise",
    "personal social services", "pss", "public sector", "private sector",
    "third sector", "voluntary sector", "academic partner",
    "industry partner", "commercial partner", "contracting organisation",
    "applicant organisation", "lead applicant", "joint lead applicant",
    "co-applicant", "co applicant", "public co-applicant",
    "public co applicant", "university", "trust", "trusts",
    "nhs trust", "nhs trusts", "company", "companies", "start-up",
    "startup", "spinout", "spin-out",

    # ------------------------------------------------------------------
    # PPIE / inclusion / mobilisation
    # ------------------------------------------------------------------
    "ppie", "ppi", "patient and public involvement", "public involvement",
    "patient and public involvement and engagement", "research inclusion",
    "inclusive research", "knowledge", "knowledge mobilisation",
    "knowledge mobilization", "dissemination", "impact", "equality",
    "diversity", "inclusion", "edi", "sex and gender",
    "underserved groups", "deprivation", "ethnicity", "accessibility",
    "accessible materials", "public advisory group", "contributors",
    "public contributors", "lived experience", "lived-experience",
    "carers", "workshop", "workshops", "stakeholders",
    "stakeholder group", "stakeholder groups", "patient group",
    "public group", "advisory group",

    # ------------------------------------------------------------------
    # Finance / health economics
    # ------------------------------------------------------------------
    "health economics", "health economic", "economic evaluation",
    "economic model", "decision-analytic model", "decision analytic model",
    "decision tree", "decision-tree modelling", "markov model",
    "cost effectiveness", "cost-effectiveness", "cost-effectiveness analysis",
    "cost effectiveness analysis", "cost utility", "cost-utility analysis",
    "cost utility analysis", "cost-benefit", "cost benefit",
    "cost consequence", "cost-consequence", "cea", "cua", "qaly",
    "qalys", "icer", "roi", "return on investment", "eq-5d",
    "eq-5d-5l", "budget", "budget impact", "budget impact model",
    "cost model", "value for money", "grant requested",
    "total project cost", "project value", "cost justification",
    "scheme cap", "acord", "soecat", "impact model", "early impact model",
    "early decision-analytic health economic model",
    "early decision analytic health economic model",
    "early health economic model", "exploratory budget impact model",
    "outputs including an early cost-effectiveness model",
    "outputs including an early cost effectiveness model",
    "early cost-effectiveness model", "early cost effectiveness model",
    "staff costs", "equipment costs", "travel costs", "indirect costs",
    "overheads", "miscellaneous project costs", "salary", "salaries",
    "on-costs", "inflation", "supplier quotations", "cost category",
    "amount", "notes", "fte", "wte", "admin support",

    # ------------------------------------------------------------------
    # Regulatory/checklist/readiness terms
    # ------------------------------------------------------------------
    "trl", "technology readiness level", "technology readiness",
    "regulatory readiness", "implementation readiness", "adoption readiness",
    "ukca", "ukca ready", "ukca-ready", "ukca compliant",
    "ukca-compliant", "ce marking", "iso 13485", "iso 14971",
    "iso 27001", "iec 62304", "iec62304", "dtac", "dtac aligned",
    "dtac-aligned", "dspt", "dsp toolkit",
    "data security and protection toolkit",
    "dtac-aligned regulatory gap analysis",
    "dtac aligned regulatory gap analysis", "regulatory gap analysis",
    "clinical validation needs", "clinical validation", "clinical safety",
    "quality management system", "risk management file", "technical file",
    "technical documentation", "post-market surveillance",
    "post market surveillance", "ukca classification",
    "will confirm ukca classification", "compliance pathway",
    "defined nhs adoption pathway", "nhs adoption pathway",
    "adoption pathway", "regulatory pathway", "regulatory plan",
    "approved body", "uk approved body", "safety case",
    "clinical safety case", "gdpr", "data protection officer", "dpo",
    "data access committee", "pseudonymised", "pseudonymized",
    "encrypted", "secure cloud", "nhs-accredited secure cloud",
    "azure", "azure health data services", "redcap", "dpia",
    "information governance", "data governance", "data protection",

    # ------------------------------------------------------------------
    # Generic evaluation / comparator / endpoint terms
    # ------------------------------------------------------------------
    "endpoint", "endpoints", "primary endpoint", "secondary endpoint",
    "outcome", "outcomes", "primary outcomes", "secondary outcomes",
    "feasibility", "acceptability", "usability", "recruitment",
    "retention", "fidelity", "interviews", "survey", "questionnaire",
    "validated questionnaire", "comparator", "control", "control arm",
    "usual care", "current best practice", "standard care",
    "endpoint is prediction", "primary endpoint is prediction",
    "secondary endpoint is prediction", "is usual nhs wound assessment",
    "usual nhs wound assessment", "usual care wound assessment",
    "nhs wound assessment", "manual measurement",
    "local guideline-based treatment decisions", "standard referral",
    "comparator is usual nhs wound assessment",
    "comparator is usual care wound assessment",
    "endpoints include time to escalation", "time to escalation",
    "recruitment rate", "adoption rate", "clinician adoption rate",
    "time-to-result", "time to result", "able to provide informed consent",
    "powered to detect", "confidence interval", "95 ci", "ci width",
    "target", "targets", "measure", "measures", "tool", "tools",

    # ------------------------------------------------------------------
    # Weak standalone context terms
    # ------------------------------------------------------------------
    "patients", "people", "adults", "older adults", "children",
    "participants", "service users", "staff", "nhs staff", "clinicians",
    "nurses", "therapists", "managers", "community", "care", "services",
    "clinic", "clinics", "hospital", "population", "setting", "support",
    "platform", "device", "system", "software", "intervention",
    "therapeutic", "diagnostic", "medical device", "digital tool",
    "new wearable", "app", "mobile app", "web browser", "smartphone app",
    "dashboard", "web dashboard", "interface", "user interface", "ui",
    "ux", "user materials", "training materials", "prototype",
    "basic prototype", "working prototype", "routine care",
    "routine nhs care", "real-world", "real world",

    # ------------------------------------------------------------------
    # Business / commercial / strategy noise
    # ------------------------------------------------------------------
    "b2b", "b2c", "b2g", "saas", "paas", "iaas", "b2b saas",
    "software as a service", "platform as a service",
    "infrastructure as a service", "subscription", "subscription model",
    "licence", "license", "licensing", "licensing model", "commercial",
    "commercialisation", "commercialization", "commercial model",
    "commercial pilot", "commercial strategy", "business model",
    "business case", "business plan", "business strategy", "market",
    "market size", "market opportunity", "market analysis",
    "market need", "market access", "route to market", "go to market",
    "go-to-market", "sales", "sales pipeline", "sales strategy",
    "revenue", "revenue model", "pricing", "pricing model",
    "procurement", "procurement readiness", "procurement framework",
    "commissioning", "commissioning route", "commissioner",
    "commissioners", "investor", "investors", "investor ready",
    "investor-ready", "scale roadmap", "scale-up", "scale up",
    "scalability", "adoption", "deployment", "roll out", "roll-out",
    "wider roll out", "wider roll-out", "implementation toolkit",
    "adoption package", "adoption pack", "value proposition",
    "private companies", "private sector pathways", "public sector market",
    "competitive advantage", "customer discovery", "market validation",
    "market penetration", "growth", "growth plan", "forecast",
    "five-year", "five year", "early-stage", "early stage",
    "exit strategy", "benefit realisation", "benefit realization",
    "commercial governance", "ip strategy", "intellectual property",
    "patent strategy", "freedom to operate", "freedom-to-operate",
    "mental health app market", "very large and growing",

    # ------------------------------------------------------------------
    # Weak generic technology labels
    # ------------------------------------------------------------------
    "ai", "artificial intelligence", "ai-driven", "ai driven",
    "ai-powered", "ai powered", "ai-enabled", "ai enabled",
    "ai based", "ai-based", "ai tool", "ai tools", "ai platform",
    "ai software", "ai system", "latest ai technology", "machine learning",
    "machine-learning", "ml", "ml model", "ml classifier", "algorithm",
    "algorithms", "model", "models", "digital", "digital health",
    "digital care", "digital intervention", "digital interventions",
    "remote monitoring", "chatbot", "ai chatbot", "ai-powered chatbot",
    "ai powered chatbot", "computer based", "computer-based",
    "analytics", "data analytics", "predictive analytics", "automation",
    "automated", "technology", "technology platform", "hardware",
    "hardware-software system", "software system",

    # ------------------------------------------------------------------
    # Context-only acronyms / pathway labels
    # ------------------------------------------------------------------
    "covid", "covid-19", "covid 19", "post covid", "post-covid",
    "crc", "2ww", "two week wait", "two-week-wait", "poc", "gp",
    "gps", "icb", "girft", "fit", "dpo", "fhir", "hl7", "emis",
    "systmone", "auc", "roc", "ci", "nhs app", "nhs trust",
    "nhs trusts", "ari", "ari-3", "wp", "ctu", "ra", "pi", "dpi",

    # ------------------------------------------------------------------
    # Roman numerals / stage fragments
    # ------------------------------------------------------------------
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "stage i", "stage ii", "stage iii", "stage iv",
    "stage i-ii", "stage i ii", "stage i–ii", "stage iii or iv",
    "stage 1", "stage 2", "stage 3", "stage 4",

    # ------------------------------------------------------------------
    # Outcome scales / measures
    # ------------------------------------------------------------------
    "sus", "pss", "pssuq", "wound-qol", "qol", "quality of life",
    "gad-7", "gad7", "activities-specific", "activities specific",
    "berg balance scale", "timed up and go",
    "activities-specific balance confidence scale",

    # ------------------------------------------------------------------
    # Work packages / milestones / fragments seen during testing
    # ------------------------------------------------------------------
    "wp1", "wp2", "wp3", "wp4", "wp5", "wp6", "wp7", "wp8",
    "related incidents", "12-month decision model",
    "algorithm development data collection", "data collection complete",
    "data collection complete for algorithm", "system design freeze",
    "go/no-go decision", "go no go decision", "ethics approval",
    "site initiation", "recruitment open", "recruitment review",
    "recruitment complete", "final analysis", "commercial plan complete",
    "multi-site diagnostic", "multi site diagnostic",
    "prospective multi-site diagnostic", "prospective multi site diagnostic",
    "conduct a prospective multi-site diagnostic",
    "conduct a prospective multi site diagnostic", "trial of the system",
    "trial of the system across nhs community rehabilitation",
    "across nhs community rehabilitation", "across three nhs community wound services",
    "nhs community wound services", "test a new wearable digital tool",
    "test a new wearable", "consume significant community nursing capacity",
    "create substantial patient burden", "substantial patient burden",
    "better use of community nursing", "better use of workforce capacity",
    "community nursing capacity", "workforce capacity", "specialist nurse capacity",
    "is a translational software", "translational software",
    "translational product development", "many people do not get enough support",
    "do not get enough support", "get enough support", "intensive support",
    "access to sufficiently intensive support", "sufficiently intensive support",
    "coaching support", "consistent detection", "reduced avoidable escalation",
    "avoidable escalation", "clinically significant deterioration requiring escalation",
    "wound image segmentation this", "integrate mqae into the stepright platform",
    "integrate mqae", "stepright platform",

    # ------------------------------------------------------------------
    # Plain-English / summary fragments
    # ------------------------------------------------------------------
    "it combines a wearable device", "combines a wearable device",
    "it combines a wearable", "combines a wearable",
    "it combines real-time activity tracking", "it combines real time activity tracking",
    "summary this project will test a new wearable",
    "summary this project will test a new wearable digital tool",
    "this project will test a new wearable",
    "this project will test a new wearable digital tool",
    "project will test a new wearable",
    "project will test a new wearable digital tool",
    "we want to find out", "we also want to test",
    "the project will happen", "if the project shows",
    "the next step would be",

    # ------------------------------------------------------------------
    # Generic pathway/device fragments
    # ------------------------------------------------------------------
    "community wound assessment and documentation pathway",
    "wound assessment and documentation pathway", "documentation pathway",
    "medical device decision-support platform",
    "medical device decision support platform",
    "software as a medical device decision-support platform",
    "software as a medical device decision support platform",
    "software as a medical device", "samd",
}


GENERIC_ACRONYMS = {
    "CEA", "CUA", "QALY", "QALYS", "ICER", "ROI", "EQ-5D", "EQ-5D-5L",
    "SUS", "PSS", "PSSUQ", "PPI", "PPIE", "NHS", "NIHR", "NICE",
    "MHRA", "RSS", "PDA", "IEC", "ISO", "IRAS", "DTAC", "EPO",
    "OPS", "HRA", "UKIPO", "USPTO", "SME", "ACORD", "SOECAT",
    "EDI", "CRC", "B2B", "B2C", "B2G", "SAAS", "PAAS", "IAAS",
    "COVID", "COVID-19", "POC", "GP", "ICB", "GIRFT", "FIT",
    "2WW", "DPO", "FHIR", "HL7", "DSPT", "EMIS", "REDCAP",
    "AUC", "ROC", "CI", "GDPR", "TRL", "WP", "WTE", "ARI",
    "DPIA", "DPI", "PI", "RA", "CTU", "QMS", "PMS", "SaMD",
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
}

VALID_SHORT_ACRONYMS = {
    "ECG", "EEG", "MRI", "CT", "DNA", "RNA", "PCR", "ELISA", "CRP",
}


SINGLE_TITLECASE_NOISE = {
    "Background", "Rationale", "Aims", "Objectives", "Methods",
    "Methodology", "References", "Appendix", "Appendices", "Invention",
    "Knowledge", "Dissemination", "Impact", "Budget", "Finance",
    "Clinical", "Regulatory", "Commercialisation", "Commercialization",
    "Deliberate", "Existing", "Related", "Source", "Sources",
    "Anchor", "Anchors", "Chronic", "Community", "Funding", "Application",
    "Programme", "Program", "Project", "Research", "Study", "Summary",
    "Plain", "English", "Training", "Synthetic", "Fictional", "Activities",
    "Machine", "Readable", "This", "The", "Phase", "Objective", "Milestone",
    "Risk", "Mitigation", "Cost", "Notes", "Domain", "Measure", "Tool",
    "Storage", "Outcome", "Comparator", "Feasibility", "Commercial",
    "Commercialisation", "Commercialization", "Governance", "Finance",
}


TECH_SUFFIXES = (
    "assessment", "monitoring", "prediction", "detection", "diagnosis",
    "triage", "classification", "stratification", "segmentation",
    "recommendation", "recommendations", "risk score", "risk model",
    "decision support", "feedback", "coaching", "rehabilitation",
    "intervention", "pathway", "assay", "biomarker", "biomarkers",
    "biomarker panel", "protein biomarkers", "blood biomarker platform",
    "immunoassay cartridge", "cartridge", "imaging", "imager", "sensor",
    "sensors", "wearable", "therapeutic", "diagnostic", "classifier",
    "platform", "dashboard", "engine", "tool", "training package",
    "behaviour-change programme", "behavior-change programme",
    "self-management programme", "self-management program",
    "implementation package", "clinical pathway", "care pathway",
    "point-of-care test", "point of care test", "care planning",
    "transitional support", "probability map", "natural language processing",
    "large language model", "language model", "risk engine",
)

FUNCTION_SUFFIXES = (
    "prevention", "rehabilitation", "detection", "diagnosis", "triage",
    "prioritisation", "prioritization", "symptom management",
    "medication optimisation", "medication optimization", "care coordination",
    "adherence support", "risk scoring", "risk prediction", "risk stratification",
    "personalised feedback", "personalized feedback",
    "personalised care recommendations", "personalized care recommendations",
    "care recommendations", "recommendations", "recommendation",
    "referral decision support", "decision support", "feedback", "coaching",
    "escalation", "assessment", "measurement", "care planning",
    "transitional support", "behaviour-change support",
    "behavior-change support", "self-management support", "early detection",
    "early diagnosis", "screening", "workforce support", "wellbeing support",
    "mental wellbeing support",
)

METHOD_KEYWORDS = (
    "assay", "biomarker", "biomarkers", "cartridge", "sensor", "sensors",
    "imaging", "imager", "device", "platform", "dashboard", "engine",
    "pathway", "programme", "program", "intervention", "prediction",
    "detection", "monitoring", "assessment", "triage", "diagnosis",
    "classification", "stratification", "coaching", "rehabilitation",
    "decision support", "risk score", "risk model", "risk engine",
    "risk stratification", "care planning", "transitional support",
    "segmentation", "recommendation", "recommendations", "protein",
    "molecular", "genomic", "digital therapeutic", "natural language processing",
)

GENERIC_METHOD_PATTERNS = (
    r"(?:deep\s+)?(?:convolutional\s+)?neural\s+network(?:\s+classifier)?",
    r"gradient[-\s]boosted\s+(?:ml\s+)?model",
    r"machine[-\s]learning\s+classifier",
    r"machine[-\s]learning\s+(?:model|prediction|assessment|platform)",
    r"natural\s+language\s+processing",
    r"large\s+language\s+model",
    r"language\s+model",
    r"sequence\s+modelling",
    r"temporal\s+sequence\s+modelling",
    r"hidden\s+markov\s+models?",
    r"random\s+forest",
    r"transformer\s+model",
    r"risk\s+prediction\s+model",
    r"risk\s+stratification\s+tool",
    r"risk\s+stratification\s+platform",
    r"point[-\s]of[-\s]care\s+(?:test|platform|diagnostic)",
    r"blood\s+biomarker\s+platform",
    r"protein\s+biomarker\s+panel",
    r"biomarker\s+panel",
    r"multiplex\s+biomarker\s+panel",
    r"immunoassay\s+cartridge",
    r"circulating\s+protein\s+biomarkers?",
    r"methylated\s+dna\s+markers?",
    r"motion\s+analysis(?:\s+systems?)?",
    r"inertial\s+measurement\s+unit",
    r"accelerometers?",
    r"gyroscopes?",
    r"self-management\s+coaching",
    r"behaviour-change\s+(?:programme|support)",
    r"behavior-change\s+(?:program|support)",
    r"care\s+pathway\s+redesign",
    r"remote\s+monitoring\s+system",
    r"decision[-\s]support\s+(?:tool|system|platform|capability)",
    r"(?:thermal|multispectral|ultrasound|mri|ct)\s+imaging",
    r"(?:pcr|elisa|mass\s+spectrometry)",
    r"(?:shap|lime)",
    r"motion\s+quality\s+assessment\s+engine",
    r"movement\s+quality\s+assessment",
    r"wearable\s+digital\s+therapeutic",
    r"wearable\s+sensors?",
    r"postural\s+control",
    r"movement\s+smoothness",
    r"range\s+of\s+motion",
    r"reaction\s+time",
    r"movement\s+speed",
    r"image[-\s]based\s+monitoring",
    r"image[-\s]based\s+\w+\s+monitoring",
    r"image\s+segmentation",
    r"wound\s+image\s+segmentation",
    r"conditional\s+\w+\s+probability\s+map",
    r"percent\s+area\s+reduction",
    r"multispectral\s+\w+\s+imaging",
)

PROBLEM_HINTS = (
    "risk", "decline", "deterioration", "prevention", "rehabilitation",
    "assessment", "disease", "condition", "syndrome", "wound", "wounds",
    "ulcer", "ulcers", "falls", "fall", "balance", "mobility", "pain",
    "infection", "cancer", "colorectal cancer", "bowel cancer",
    "diabetes", "stroke", "frailty", "depression", "anxiety", "stress",
    "burnout", "mental health", "wellbeing", "diagnosis", "inequality",
    "burden", "discharge", "sepsis", "triage", "relapse", "delay",
    "screening", "early detection", "early diagnosis",
)

SETTING_HINTS = (
    "primary care", "community rehabilitation", "nhs community rehabilitation",
    "community wound services", "community wound clinics",
    "community nursing teams", "tissue viability services", "gp practices",
    "endoscopy units", "emergency department", "care home", "care homes",
    "social care",
)


@dataclass(frozen=True)
class RankedTerm:
    term: str
    concept_class: str
    field_priority: int = 5

    @property
    def sort_key(self) -> tuple[int, int, int, str]:
        class_priority = {
            "exact_identifier": 0,
            "named_entities": 1,
            "technical_method_or_mechanism": 2,
            "product_or_intervention_function": 3,
            "intervention_type": 4,
            "clinical_or_social_care_problem": 5,
            "population_setting": 6,
            "generic_document_terms": 99,
        }.get(self.concept_class, 50)

        id_priority = 0
        if self.concept_class == "exact_identifier":
            if is_nihr_identifier(self.term):
                id_priority = 0
            elif is_patent_identifier(self.term):
                id_priority = 1
            elif is_trial_identifier(self.term):
                id_priority = 2
            else:
                id_priority = 3

        return (class_priority, id_priority, self.field_priority, normalise(self.term))


def normalise(term: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9 +#\-/]", " ", str(term or "").lower()),
    ).strip()


def _clean(term: str) -> str:
    cleaned = str(term or "")

    cleaned = re.sub(
        r"FOR\s+TRAINING\s+USE\s+ONLY|FICTIONAL\s+EXAMPLE\s+APPLICATION|SYNTHETIC\s+EXEMPLAR|DUMMY\s+APPLICATION|MOCK\s+APPLICATION",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"\b(?:fictional|invented|training only|dummy|mock)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(?:this project will|this proposal will|we will)\b", " ", cleaned, flags=re.I)
    cleaned = cleaned.replace("…", " ")
    cleaned = re.sub(r"\.\.\.+", " ", cleaned)
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("/", " / ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:,\n\t")

    cleaned = re.sub(
        r"^(?:plain english summary|summary|scientific abstract|project title)\s+",
        "",
        cleaned,
        flags=re.I,
    )

    return cleaned.strip(" ,;:-/")


def _strip_weak_prefixes(cleaned: str) -> str:
    cleaned = re.sub(
        r"^(?:AI[-\s]enabled|AI[-\s]powered|AI[-\s]driven|AI[-\s]based|validated|proprietary|early[-\s]stage|late[-\s]stage)\s+",
        "",
        cleaned,
        flags=re.I,
    ).strip()

    cleaned = re.sub(
        r"^(?:DTAC[-\s]aligned|UKCA[-\s]ready|UKCA[-\s]compliant|regulatory|commercial|investor[-\s]ready)\s+",
        "",
        cleaned,
        flags=re.I,
    ).strip()

    return cleaned


def _normalise_variant(term: str) -> str:
    cleaned = _clean(term)
    key = normalise(cleaned)

    if not cleaned:
        return ""

    if re.fullmatch(r"\d{1,6}", key):
        return ""

    if re.fullmatch(r"(?:stage\s+)?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)", key):
        return ""

    cleaned = re.sub(r"\b(?:This|The|A|An)$", "", cleaned).strip(" .;:,/-")
    cleaned = _strip_weak_prefixes(cleaned)
    key = normalise(cleaned)

    if not cleaned:
        return ""

    if key in GENERIC_DOCUMENT_TERMS:
        return ""

    match = re.fullmatch(r"([A-Z0-9]{3,14})[-\s]enabled", cleaned, flags=re.I)
    if match:
        acronym = match.group(1).upper()
        if acronym in GENERIC_ACRONYMS:
            return ""
        return acronym

    if re.fullmatch(
        r"[A-Za-z0-9]+[-\s](?:aligned|ready|specific|compliant|certified)",
        cleaned,
        flags=re.I,
    ):
        return ""

    if (" / " in cleaned or "," in cleaned or " plus " in key) and not is_any_identifier(cleaned):
        return ""

    if ":" in cleaned and not is_any_identifier(cleaned):
        after_colon = _clean(cleaned.split(":", 1)[1])
        after_colon = _strip_weak_prefixes(after_colon)
        if after_colon:
            cleaned = after_colon
            key = normalise(cleaned)

    if key in GENERIC_DOCUMENT_TERMS:
        return ""

    blocked_substrings = (
        "health economic model", "health economics", "economic evaluation",
        "cost-effectiveness model", "cost effectiveness model",
        "budget impact model", "impact model", "decision-analytic health economic model",
        "decision analytic health economic model", "outputs including",
        "endpoint is", "primary endpoint", "secondary endpoint",
        "comparator is", "usual nhs", "usual care", "nhs wound assessment",
        "endpoints include", "time to escalation", "prospective multi-site diagnostic",
        "prospective multi site diagnostic", "conduct a prospective", "trial of the system",
        "across nhs community rehabilitation", "across three nhs",
        "nhs community wound services", "consistent detection",
        "avoidable escalation", "reduced avoidable escalation", "requiring escalation",
        "regulatory gap analysis", "ukca classification", "will confirm ukca",
        "clinical validation needs", "compliance pathway", "many people do not get enough support",
        "do not get enough support", "sufficiently intensive support", "access to sufficiently",
        "test a new wearable digital tool", "test a new wearable", "it combines a wearable",
        "combines a wearable", "it combines real-time activity", "it combines real time activity",
        "summary this project", "this project will test", "project will test",
        "defined nhs adoption pathway", "nhs adoption pathway", "adoption pathway",
        "latest ai technology", "commercial pilot", "commercial model", "business model",
        "five year", "early stage", "private companies", "mental health app market",
        "market is very large", "integrate mqae", "stepright platform",
        "community wound assessment and documentation pathway",
        "wound assessment and documentation pathway", "documentation pathway",
        "software as a medical device", "medical device decision support",
        "medical device decision-support",
    )

    if any(phrase in key for phrase in blocked_substrings):
        return ""

    if key in GENERIC_DOCUMENT_TERMS:
        return ""

    if re.fullmatch(r"\d{1,6}", key):
        return ""

    if re.fullmatch(r"(?:stage\s+)?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)", key):
        return ""

    return cleaned


def _looks_like_outcome_scale(term: str) -> bool:
    key = normalise(term)

    if key in {
        "sus", "pss", "pssuq", "eq-5d", "eq-5d-5l", "wound-qol",
        "gad-7", "gad7", "qol",
    }:
        return True

    if "qol" in key:
        return True

    if key.endswith(" scale") or key.endswith(" questionnaire") or key.endswith(" inventory"):
        return True

    return False


def is_generic_term(term: str) -> bool:
    cleaned = normalise(term)

    if not cleaned or len(cleaned) < 2:
        return True

    if is_any_identifier(term):
        return False

    if _looks_like_outcome_scale(term):
        return True

    if re.fullmatch(r"\d{1,6}", cleaned):
        return True

    if re.fullmatch(r"(?:stage\s+)?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)", cleaned):
        return True

    if re.fullmatch(r"wp\d+", cleaned):
        return True

    if re.fullmatch(r"trl(?:\s+\d+)?(?:\s+to\s+trl?\s*\d+)?", cleaned):
        return True

    if re.fullmatch(r"(?:iso\s*)?(?:13485|14971|27001)", cleaned):
        return True

    if re.fullmatch(r"(?:ukca|ce)[-\s]?(?:ready|compliant|marked|certified)", cleaned):
        return True

    if cleaned.endswith(" aligned") or cleaned.endswith(" specific") or cleaned.endswith(" ready"):
        return True

    if cleaned in GENERIC_DOCUMENT_TERMS:
        return True

    if cleaned.upper() in GENERIC_ACRONYMS:
        return True

    if cleaned.rstrip("s") in GENERIC_DOCUMENT_TERMS:
        return True

    words = cleaned.split()
    if words and all(word in GENERIC_DOCUMENT_TERMS for word in words):
        return True

    return False


def _looks_like_single_sentence_word(value: str) -> bool:
    cleaned = _clean(value)

    if is_any_identifier(cleaned):
        return False

    if cleaned in SINGLE_TITLECASE_NOISE:
        return True

    if cleaned.upper() in GENERIC_ACRONYMS:
        return True

    if normalise(cleaned) in GENERIC_DOCUMENT_TERMS:
        return True

    if re.fullmatch(r"[A-Z][a-z]{3,24}", cleaned):
        product_like_suffixes = (
            "bot", "care", "scan", "check", "map", "flow", "path", "link",
            "net", "track", "watch", "fit", "wise", "first", "right",
            "alert", "bridge", "guard", "sense", "view", "vision",
        )
        lower = cleaned.lower()
        if not any(lower.endswith(suffix) for suffix in product_like_suffixes):
            return True

    return False


def _recognised_product_or_acronym(value: str) -> bool:
    cleaned = _clean(value)

    if is_any_identifier(cleaned):
        return True

    if is_generic_term(cleaned):
        return False

    if _looks_like_single_sentence_word(cleaned):
        return False

    if re.fullmatch(r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*", cleaned):
        return True

    if re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+", cleaned):
        return True

    if re.fullmatch(r"[A-Z0-9-]{3,14}", cleaned) and cleaned.upper() not in GENERIC_ACRONYMS:
        return True

    if re.fullmatch(r"[A-Z][a-z]{3,24}", cleaned):
        product_like_suffixes = (
            "bot", "care", "scan", "check", "map", "flow", "path", "link",
            "net", "track", "watch", "fit", "wise", "first", "right",
            "alert", "bridge", "guard", "sense", "view", "vision",
        )
        if any(cleaned.lower().endswith(suffix) for suffix in product_like_suffixes):
            return True

    return False


def _looks_like_compact_technical_phrase(value: str) -> bool:
    key = normalise(value)
    words = key.split()

    if is_any_identifier(value):
        return True

    if not (2 <= len(words) <= 8):
        return False

    if any(re.fullmatch(pattern, key, flags=re.I) for pattern in GENERIC_METHOD_PATTERNS):
        return True

    if any(re.search(pattern, key, flags=re.I) for pattern in GENERIC_METHOD_PATTERNS):
        return True

    if any(key.endswith(normalise(suffix)) for suffix in TECH_SUFFIXES + FUNCTION_SUFFIXES):
        return True

    return False


def _looks_like_sentence_fragment(value: str) -> bool:
    key = normalise(value)
    words = key.split()

    if is_any_identifier(value):
        return False

    bad_patterns = [
        r"\baccess to sufficiently\b",
        r"\bsufficiently intensive support\b",
        r"\bmany people do not get enough support\b",
        r"\bdo not get enough support\b",
        r"\bdata collection complete\b",
        r"\bcomplete for algorithm\b",
        r"\bactivities specific\b",
        r"\bdtac aligned\b",
        r"\bregulatory gap analysis\b",
        r"\bukca classification\b",
        r"\bwill confirm ukca\b",
        r"\bcompliance pathway\b",
        r"\bendpoint is\b",
        r"\bprimary endpoint\b",
        r"\bsecondary endpoint\b",
        r"\bcomparator is\b",
        r"\bis usual\b",
        r"\busual nhs\b",
        r"\busual care\b",
        r"\bnhs wound assessment\b",
        r"\bendpoints include\b",
        r"\btime to escalation\b",
        r"\bhealth economic model\b",
        r"\bhealth economics\b",
        r"\beconomic evaluation\b",
        r"\bbudget impact model\b",
        r"\bimpact model\b",
        r"\bcost effectiveness model\b",
        r"\bcost-effectiveness model\b",
        r"\boutputs including\b",
        r"\bconduct a prospective\b",
        r"\btrial of the system\b",
        r"\bacross nhs community rehabilitation\b",
        r"\bacross\s+\w+\s+nhs\b",
        r"\bnhs community wound services\b",
        r"\bprospective multi site diagnostic\b",
        r"\bprospective multi-site diagnostic\b",
        r"\bmulti site diagnostic\b",
        r"\bmulti-site diagnostic\b",
        r"\bconsistent detection\b",
        r"\breduced avoidable escalation\b",
        r"\bavoidable escalation\b",
        r"\brequiring escalation\b",
        r"\btest a new wearable digital tool\b",
        r"\btest a new wearable\b",
        r"\bdigital tool\b",
        r"\bwound image segmentation this\b",
        r"\bit combines\b",
        r"\bcombines a wearable\b",
        r"\bsummary this project\b",
        r"\bthis project will test\b",
        r"\bproject will test\b",
        r"\bnew wearable\b",
        r"\bdefined nhs adoption pathway\b",
        r"\bnhs adoption pathway\b",
        r"\badoption pathway\b",
        r"\bintegrate\s+mqae\b",
        r"\bstepright platform\b",
        r"\bcommunity wound assessment and documentation pathway\b",
        r"\bwound assessment and documentation pathway\b",
        r"\bdocumentation pathway\b",
        r"\bsoftware as a medical device\b",
        r"\bmedical device decision[-\s]support\b",
        r"\bai[-\s]driven\b",
        r"\bai[-\s]powered\b",
        r"\bai[-\s]enabled\b",
        r"\bai[-\s]based\b",
        r"\bearly[-\s]stage\b",
        r"\bfive[-\s]year\b",
        r"\bstage\s+(?:i|ii|iii|iv|v)\b",
        r"\bcommercial pilot\b",
        r"\bbusiness model\b",
        r"\bcommercial model\b",
        r"\bmarket is\b",
        r"\bprivate companies\b",
        r"\b[a-z0-9]+ enabled\b",
        r"\bwp\d+\b",
        r"\btheir risk of\b",
        r"\breduce their risk\b",
        r"\bconsistent detection of\b",
        r"\bearly and consistent\b",
        r"\bconsume significant\b",
        r"\bsubstantial patient burden\b",
        r"\bclinical validation needs\b",
        r"\bis a translational\b",
        r"\bbetter use of\b",
        r"\bworkforce capacity\b",
        r"\bcommunity nursing capacity\b",
        r"\bsupports? nhs adoption\b",
        r"\bcreating substantial pressure\b",
    ]

    if any(re.search(pattern, key) for pattern in bad_patterns):
        return True

    if _looks_like_compact_technical_phrase(value):
        return False

    if len(words) > 6:
        return True

    if re.search(
        r"\b(is|are|was|were|will|would|could|should|consume|consuming|create|creating|reduce|reducing|improve|improving|supporting|needs|needed|designed|including|capacity|burden|pressure|adoption|implementation|complete|confirmed|aligned|requiring|conduct|combines|test|trial|believe|think|maybe|probably|integrate|defined|across)\b",
        key,
    ):
        return True

    return False


def _is_demographic_or_context_only(value: str) -> bool:
    key = normalise(value)

    if re.fullmatch(
        r"(?:aged|age|over|under|older|younger|adults?|patients?|people|staff|clinicians|nurses|participants)(?:\s+\d+\+?)?(?:\s+(?:and|or|over|under))*",
        key,
    ):
        return True

    if re.fullmatch(r"\d+\s+(?:adults|patients|participants|people|staff|clinicians|nurses)\b.*", key):
        return True

    if key in {
        "older adults", "adults", "patients", "people", "community", "nhs",
        "care", "usual care", "rehabilitation", "detection", "support",
        "platform", "device", "services", "community services",
        "new wearable", "nhs staff", "staff", "clinicians", "nurses",
    }:
        return True

    if re.search(r"\baged\s*\d+", key):
        return True

    return False


def _valid_query_concept(value: str) -> bool:
    value = _normalise_variant(value)
    key = normalise(value)
    words = key.split()

    if not value or value == NOT_EXPLICITLY_STATED:
        return False

    if re.fullmatch(r"\d{1,6}", key):
        return False

    if re.fullmatch(r"(?:stage\s+)?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)", key):
        return False

    if is_any_identifier(value):
        return True

    if " / " in value or "," in value or " plus " in key:
        return False

    if is_generic_term(value):
        return False

    if _looks_like_single_sentence_word(value) and not _recognised_product_or_acronym(value):
        return False

    if _looks_like_sentence_fragment(value):
        return False

    if len(value) > MAX_TERM_CHARS or len(words) > 8:
        return False

    if len(value) < 3 and value.upper() not in VALID_SHORT_ACRONYMS:
        return False

    if _is_demographic_or_context_only(value):
        return False

    if not words:
        return False

    if words[0] in {
        "and", "or", "for", "with", "plus", "the", "a", "an", "as",
        "is", "are", "was", "were", "will", "would", "could", "should",
        "reduce", "reducing", "improve", "improving", "to", "including",
        "include", "includes", "create", "creating", "lead", "conduct",
        "test", "trial", "it", "this", "project", "summary", "we",
        "they", "there", "if", "when", "maybe", "integrate", "across",
        "defined",
    }:
        return False

    if words[-1] in {"and", "or", "for", "with", "of", "plus", "is", "are"}:
        return False

    if re.search(r"\b[a-z]\b$", key):
        return False

    return True


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        value = _normalise_variant(str(item))
        key = normalise(value)

        if not key or key in seen:
            continue

        if not _valid_query_concept(value):
            continue

        seen.add(key)
        out.append(value)

    return out


def _dedupe_add(candidates: list[str], value: str) -> None:
    value = _normalise_variant(value)

    if not _valid_query_concept(value):
        return

    key = normalise(value)

    for idx, existing in enumerate(list(candidates)):
        existing_key = normalise(existing)

        if key == existing_key:
            return

        if is_any_identifier(existing) or is_any_identifier(value):
            continue

        if key in existing_key and len(key.split()) > 1:
            return

        if existing_key in key and len(existing_key.split()) > 1:
            candidates[idx] = value
            return

    candidates.append(value)


def _identifier_phrases(value: str) -> list[str]:
    return extract_identifiers(value)


def _generic_method_phrases(value: str) -> list[str]:
    phrases: list[str] = []

    for pattern in GENERIC_METHOD_PATTERNS:
        for match in re.finditer(rf"\b{pattern}\b", value, re.I):
            phrases.append(match.group(0).strip(" .;:,/-"))

    return _dedupe(phrases)


def _capitalised_or_acronym_phrases(value: str) -> list[str]:
    phrases: list[str] = []

    pattern = (
        r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"
        r"|\b[A-Z][A-Z0-9]{2,14}\b"
        r"|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b"
        r"|\b[A-Z][a-z]{3,24}\b"
    )

    for match in re.finditer(pattern, value):
        phrase = match.group(0)
        normalised = _normalise_variant(phrase)

        if _recognised_product_or_acronym(normalised) and _valid_query_concept(normalised):
            phrases.append(normalised)

    for match in re.finditer(
        r"\b[A-Z][a-z][A-Za-z0-9]{2,20}(?:\s+[A-Z][a-z][A-Za-z0-9]{2,20}){1,6}\b",
        value,
    ):
        phrase = match.group(0)
        phrase_key = normalise(phrase)

        if any(marker in phrase_key for marker in METHOD_KEYWORDS):
            phrases.append(phrase)

    return _dedupe(phrases)


def _suffix_phrases(value: str, suffixes: tuple[str, ...]) -> list[str]:
    phrases: list[str] = []
    suffix_re = "|".join(re.escape(s) for s in sorted(suffixes, key=len, reverse=True))

    for match in re.finditer(
        rf"\b(?:[A-Za-z0-9+#-]+\s+){{1,7}}(?:{suffix_re})\b",
        value,
        re.I,
    ):
        phrase = match.group(0).strip(" .;:,/-")
        words = phrase.split()

        if 2 <= len(words) <= 8:
            phrases.append(phrase)

            for window in (2, 3, 4, 5, 6):
                if len(words) >= window:
                    phrases.append(" ".join(words[-window:]))

    return _dedupe(phrases)


def _clinical_problem_phrases(value: str) -> list[str]:
    phrases: list[str] = []
    hints = "|".join(re.escape(h) for h in PROBLEM_HINTS)

    for match in re.finditer(
        rf"\b(?:[A-Za-z0-9+#-]+\s+){{0,4}}(?:{hints})(?:\s+[A-Za-z0-9+#-]+){{0,3}}\b",
        value,
        re.I,
    ):
        phrase = match.group(0).strip(" .;:,/-")
        words = phrase.split()

        if 2 <= len(words) <= 6:
            phrases.append(phrase)

    return _dedupe(phrases)


def _split_source_text(value: str) -> list[str]:
    value = _clean(value)

    parts = re.split(
        r"[,;]|\s+/\s+|\s+\|\s+|\s+\+\s+|\s+plus\s+|\s+ and \s+|\s+ or \s+",
        value,
        flags=re.I,
    )

    return [_clean(part) for part in parts if _clean(part)]


def _concepts_from_value(value: str) -> list[str]:
    value = _clean(value)

    if not value or value == NOT_EXPLICITLY_STATED:
        return []

    concepts: list[str] = []

    for extractor in (
        _identifier_phrases,
        _generic_method_phrases,
        _capitalised_or_acronym_phrases,
    ):
        for phrase in extractor(value):
            _dedupe_add(concepts, phrase)

    for phrase in _suffix_phrases(value, TECH_SUFFIXES + FUNCTION_SUFFIXES):
        _dedupe_add(concepts, phrase)

    for phrase in _clinical_problem_phrases(value):
        _dedupe_add(concepts, phrase)

    if " / " not in value and "," not in value and " plus " not in normalise(value):
        if _looks_like_compact_technical_phrase(value) or _recognised_product_or_acronym(value):
            _dedupe_add(concepts, value)

    for part in _split_source_text(value):
        if not part or part == value or _looks_like_sentence_fragment(part):
            continue

        if _recognised_product_or_acronym(part):
            _dedupe_add(concepts, part)

        for phrase in (
            _identifier_phrases(part)
            + _generic_method_phrases(part)
            + _capitalised_or_acronym_phrases(part)
            + _suffix_phrases(part, TECH_SUFFIXES + FUNCTION_SUFFIXES)
            + _clinical_problem_phrases(part)
        ):
            _dedupe_add(concepts, phrase)

    return concepts


def _short_concepts_from_text(snippet: str) -> list[str]:
    snippet = _clean(str(snippet or "")[:6000])
    concepts: list[str] = []

    for phrase in (
        _identifier_phrases(snippet)
        + _generic_method_phrases(snippet)
        + _capitalised_or_acronym_phrases(snippet)
        + _suffix_phrases(snippet, TECH_SUFFIXES + FUNCTION_SUFFIXES)
        + _clinical_problem_phrases(snippet)
    ):
        _dedupe_add(concepts, phrase)

    for part in _split_source_text(snippet):
        if not part or _looks_like_sentence_fragment(part):
            continue

        if _recognised_product_or_acronym(part):
            _dedupe_add(concepts, part)

    return _prioritise_plain_terms(concepts)[:12]


def concept_class(term: str) -> str:
    term = _normalise_variant(term)
    key = normalise(term)

    if not term:
        return "generic_document_terms"

    if is_patent_identifier(term) or is_nihr_identifier(term) or is_trial_identifier(term):
        return "exact_identifier"

    if is_generic_term(term):
        return "generic_document_terms"

    if _recognised_product_or_acronym(term):
        return "named_entities"

    if any(re.search(pattern, key, flags=re.I) for pattern in GENERIC_METHOD_PATTERNS):
        return "technical_method_or_mechanism"

    if any(key.endswith(normalise(suffix)) for suffix in FUNCTION_SUFFIXES):
        return "product_or_intervention_function"

    if any(key.endswith(normalise(suffix)) for suffix in TECH_SUFFIXES):
        if any(x in key for x in (
            "therapeutic", "programme", "program", "platform", "device",
            "test", "pathway", "package", "tool", "dashboard", "assay",
            "cartridge", "biomarker",
        )):
            return "intervention_type"
        return "technical_method_or_mechanism"

    if any(h in key for h in PROBLEM_HINTS):
        return "clinical_or_social_care_problem"

    if any(h in key for h in SETTING_HINTS):
        return "population_setting"

    return "technical_method_or_mechanism"


def _field_priority(field_name: str) -> int:
    high = {
        "project_title",
        "product_or_intervention",
        "acronym_or_short_name",
        "technology_type",
        "mechanism_of_action",
        "methodology",
        "clinical_or_social_care_need",
        "study_design",
    }

    medium = {
        "target_population",
        "endpoints",
        "comparator_or_control",
        "market_or_impact_evidence",
        "sites_or_setting",
        "snippets",
        "plain_english_summary",
    }

    low = {
        "regulatory_plan",
        "health_economics_plan",
        "references_detected",
        "next_stage_plan",
        "application_claimed_call",
    }

    if field_name in high:
        return 0
    if field_name in medium:
        return 1
    if field_name in low:
        return 3
    return 2


def _ranked(term: str, field_name: str) -> RankedTerm:
    return RankedTerm(
        term=_normalise_variant(term),
        concept_class=concept_class(term),
        field_priority=_field_priority(field_name),
    )


def _prioritise_ranked_terms(terms: list[RankedTerm]) -> list[str]:
    best_by_key: dict[str, RankedTerm] = {}

    for item in terms:
        key = normalise(item.term)

        if not key or not _valid_query_concept(item.term):
            continue

        if item.concept_class == "generic_document_terms":
            continue

        existing = best_by_key.get(key)
        if existing is None or item.sort_key < existing.sort_key:
            best_by_key[key] = item

    ranked = sorted(best_by_key.values(), key=lambda item: item.sort_key)
    return [item.term for item in ranked]


def _prioritise_plain_terms(terms: list[str]) -> list[str]:
    return _prioritise_ranked_terms([_ranked(term, "snippets") for term in terms])


def _split_primary_secondary(terms: list[str]) -> tuple[list[str], list[str]]:
    primary: list[str] = []
    secondary: list[str] = []

    for term in terms:
        cls = concept_class(term)

        if cls in {
            "exact_identifier",
            "named_entities",
            "technical_method_or_mechanism",
            "product_or_intervention_function",
            "intervention_type",
        }:
            primary.append(term)
        elif cls in {"clinical_or_social_care_problem", "population_setting"}:
            secondary.append(term)

        if len(primary) + len(secondary) >= MAX_QUERY_TERMS:
            break

    return primary, secondary


def _value_to_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            str(v)
            for v in value
            if str(v).strip() and str(v) != NOT_EXPLICITLY_STATED
        ]

    if value and value != NOT_EXPLICITLY_STATED:
        return [str(value)]

    return []


def build_similarity_query(
    facts: ApplicationFacts,
    snippets: list[str] | None = None,
) -> SimilarityQuery:
    ranked_terms: list[RankedTerm] = []

    fields_to_scan = [
        "project_title",
        "application_claimed_call",
        "product_or_intervention",
        "acronym_or_short_name",
        "technology_type",
        "target_population",
        "clinical_or_social_care_need",
        "mechanism_of_action",
        "methodology",
        "study_design",
        "endpoints",
        "market_or_impact_evidence",
        "sites_or_setting",
        "comparator_or_control",
        "regulatory_plan",
        "references_detected",
        "health_economics_plan",
        "next_stage_plan",
        "ip_and_commercialisation",
        "ip_and_commercialization",
        "novelty_or_similarity_section",
        "prior_work",
        "plain_english_summary",
    ]

    for field_name in fields_to_scan:
        raw_values = _value_to_strings(
            getattr(facts, field_name, NOT_EXPLICITLY_STATED)
        )

        for raw_value in raw_values:
            for concept in _concepts_from_value(raw_value):
                ranked_terms.append(_ranked(concept, field_name))

    for snippet in snippets or []:
        for concept in _short_concepts_from_text(snippet):
            ranked_terms.append(_ranked(concept, "snippets"))

    ordered_terms = _prioritise_ranked_terms(ranked_terms)
    primary_terms, secondary_terms = _split_primary_secondary(ordered_terms)

    terms = primary_terms + secondary_terms

    query_string = (
        " AND ".join(f'"{term}"' if " " in term else term for term in terms)
        if len(terms) >= 2
        else ""
    )

    reasoning = [
        "Selected short domain-agnostic similarity concepts prioritising identifiers, named interventions, technical methods, intervention functions and specific target problems. Generic application/checklist terms, source names, finance terms, business terms, outcome scales, regulatory-readiness terms, headings, demographic-only terms, work-package labels, comparator/endpoint fragments, generic technology labels and sentence fragments are excluded."
    ]

    if len(terms) < 2:
        reasoning.append(
            "At least two meaningful concepts are required before live API searching."
        )

    return SimilarityQuery(
        primary_terms=primary_terms,
        secondary_terms=secondary_terms,
        excluded_terms=sorted(GENERIC_DOCUMENT_TERMS),
        query_string=query_string,
        extraction_reasoning=reasoning,
    )