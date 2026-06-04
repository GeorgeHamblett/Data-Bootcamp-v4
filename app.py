"""Streamlit entrypoint for the RSS/NIHR Funding Application Checklist Assistant."""
from __future__ import annotations

from time import sleep

from application_facts import extract_application_facts
from checklist_engine import build_checklist
from document_loader import combine_pasted_and_uploaded
from guidance_loader import build_baseline_requirement_bank, detects_pda_relevance
from guidance_parser import parse_guidance_text
from rag_dashboard import build_rag_dashboard
from report_renderer import (
    checklist_table_rows,
    dashboard_table_rows,
    raw_json_payload,
    render_checklist_report_summary,
    render_executive_review_note,
    render_main_case_summary,
    render_priority_missing_evidence,
    render_rag_dashboard_summary,
    render_raw_json_note,
    render_similarity_check_summary,
    render_table_display_dataframe,
    similarity_query_terms_display,
    similarity_table_rows,
)
from settings import Settings
from similarity.service import run_similarity_service
from similarity.epo_ops import check_epo_credentials

APP_TITLE = "RSS/NIHR Funding Application Checklist Assistant"
NO_SPECIFIC_CALL_GUIDANCE_MESSAGE = "No specific funding call guidance provided; review uses built-in NIHR domestic guidance and RSS PDA playbook guidance."
GENERATION_PROGRESS_STEP_DELAY_SECONDS = 0.08


def _runtime_guidance_from_inputs(pasted: str, uploads) -> str:
    docs = combine_pasted_and_uploaded(pasted, uploads)
    return "\n\n".join(doc.text for doc in docs)


def _advance_generation_progress(progress, start_percent: int, target_percent: int, delay_seconds: float) -> int:
    """Advance the progress bar gradually so users can see each stage unfold."""

    if delay_seconds <= 0 or target_percent <= start_percent:
        progress.progress(target_percent)
        return target_percent

    for value in range(start_percent + 1, target_percent + 1):
        progress.progress(value)
        sleep(delay_seconds)
    return target_percent


def _set_generation_stage(stage_display, message: str) -> None:
    """Replace the visible generation stage instead of appending a running list."""

    stage_display.markdown(f"**{message}**")


def _update_generation_progress(
    stage_display,
    progress,
    message: str,
    percent: int,
    current_percent: int,
    delay_seconds: float = GENERATION_PROGRESS_STEP_DELAY_SECONDS,
) -> int:
    """Replace the visible generation stage and advance the progress bar."""

    _set_generation_stage(stage_display, message)
    return _advance_generation_progress(progress, current_percent, percent, delay_seconds)


def _similarity_progress_message(run_similarity: bool, mock_similarity: bool, settings: Settings) -> str:
    if not run_similarity:
        return "Similarity/novelty check skipped because it was not enabled."
    if mock_similarity:
        return "Similarity/novelty check running in mock mode for testing; no live external searches are made."
    if settings.strict_local_only_mode:
        return "Similarity/novelty check blocked by strict local-only mode; no external searches are made."
    if not settings.allow_external_similarity_queries:
        return "Similarity/novelty check blocked by privacy settings; no external searches are made."
    if not settings.send_only_safe_query_terms:
        return "Similarity/novelty check blocked because the safe-query-term privacy gate is disabled."
    return "Similarity/novelty check running live with cleaned concept terms only; no full application text is sent externally."


def run_generation_pipeline(
    *,
    app_text: str,
    app_uploads,
    call_text: str,
    call_uploads,
    settings: Settings,
    run_similarity: bool,
    mock_similarity: bool,
    stage_display,
    progress,
    progress_delay_seconds: float = GENERATION_PROGRESS_STEP_DELAY_SECONDS,
) -> dict[str, object]:
    """Run report generation in visible, client-friendly stages."""

    current_progress = 0
    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "1. Reading application documents — combining pasted text and uploaded files.",
        8,
        current_progress,
        progress_delay_seconds,
    )
    application_docs = combine_pasted_and_uploaded(app_text, app_uploads)
    if not application_docs:
        raise ValueError("The Application is required. Paste text or upload .docx, .pdf, .txt or .xlsx files.")
    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "1. Reading application documents — application documents found and ready for the checklist evidence review.",
        12,
        current_progress,
        progress_delay_seconds,
    )

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "2. Extracting application facts — identifying title, applicant, intervention/product, population, study design, TRL/stage, sample size, endpoints, PPIE, inclusion, health economics, regulatory plan, work packages and uncertainties.",
        20,
        current_progress,
        progress_delay_seconds,
    )
    facts = extract_application_facts(application_docs)

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "3. Reading optional funding call guidance — combining pasted guidance and uploaded call documents.",
        32,
        current_progress,
        progress_delay_seconds,
    )
    specific_text = _runtime_guidance_from_inputs(call_text, call_uploads)
    if specific_text.strip():
        _set_generation_stage(stage_display, "3. Specific funding call guidance found and added to the built-in NIHR/RSS guidance.")
    else:
        _set_generation_stage(stage_display, "3. No specific funding call guidance supplied; using built-in NIHR/RSS guidance where relevant.")

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "4. Selecting relevant guidance — checking PDA/RSS playbook relevance and building the baseline requirement bank.",
        44,
        current_progress,
        progress_delay_seconds,
    )
    include_pda = detects_pda_relevance(
        specific_text,
        facts.application_claimed_call,
        facts.product_or_intervention,
        facts.technology_type,
        facts.trl_evidence,
    )
    baseline = build_baseline_requirement_bank(".", include_pda_playbook=include_pda)
    specific_reqs = parse_guidance_text(specific_text, "specific_call", prefix="CALL") if specific_text.strip() else []
    for req in specific_reqs:
        req.overrides_general_guidance = True

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "5. Building the checklist review — matching application evidence to requirements, flagging missing/weak/contradictory evidence and applying hard validation rules.",
        58,
        current_progress,
        progress_delay_seconds,
    )
    checklist = build_checklist(facts, baseline, specific_reqs)

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "6. Building the RAG dashboard — summarising Red/Amber/Green performance by review area and collecting validation warnings.",
        70,
        current_progress,
        progress_delay_seconds,
    )
    dashboard = build_rag_dashboard(checklist, facts)

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "7. Prioritising missing evidence — identifying the highest-priority gaps and recommended next actions.",
        80,
        current_progress,
        progress_delay_seconds,
    )
    priority = render_priority_missing_evidence(checklist, dashboard, facts)

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        f"8. Running similarity/novelty check — {_similarity_progress_message(run_similarity, mock_similarity, settings)}",
        88,
        current_progress,
        progress_delay_seconds,
    )
    similarity = run_similarity_service(
        facts,
        settings,
        run_similarity_check=run_similarity,
        mock_mode=mock_similarity,
        snippets=[doc.text[:800] for doc in application_docs],
    )

    current_progress = _update_generation_progress(
        stage_display,
        progress,
        "9. Preparing report tabs — Summary, Checklist Report, RAG Dashboard, Similarity Check, Priority Missing Evidence and Raw JSON.",
        96,
        current_progress,
        progress_delay_seconds,
    )

    return {
        "application_docs": application_docs,
        "facts": facts,
        "specific_reqs": specific_reqs,
        "baseline": baseline,
        "checklist": checklist,
        "dashboard": dashboard,
        "priority": priority,
        "similarity": similarity,
    }


def main() -> None:
    import streamlit as st
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Checklist-first adviser tool grounded in runtime application evidence and built-in NIHR/RSS guidance.")

    settings = Settings.from_env()
    with st.sidebar:
        st.header("The Application")
        st.caption("Required. This is the actual applicant submission, not built-in guidance.")
        app_text = st.text_area("Paste application text", height=220, help="Required unless application files are uploaded.")
        app_uploads = st.file_uploader("Upload application/supporting documents", type=["docx", "pdf", "txt", "xlsx"], accept_multiple_files=True)
        st.header("Optional Specific Funding Call Guidance")
        st.caption("Optional but recommended when exact call page/opportunity guidance is available.")
        call_text = st.text_area("Paste specific funding call guidance", height=130)
        call_uploads = st.file_uploader("Upload specific funding call guidance", type=["docx", "pdf", "txt"], accept_multiple_files=True)
        st.subheader("Similarity settings")
        run_similarity = st.checkbox("Run similarity check", value=False)
        with st.expander("Advanced developer/testing options"):
            mock_similarity = st.checkbox("Mock similarity mode", value=False)
            show_raw_requirements = st.checkbox("Show raw extracted requirements", value=False)
            st.write("Credential status", settings.credential_status())
            if st.button("Test EPO OPS credentials"):
                epo_status = check_epo_credentials(settings)
                if epo_status.get("status") == "success":
                    st.success(epo_status.get("message", "EPO OPS authentication succeeded."))
                else:
                    st.warning(epo_status.get("why_relevant") or epo_status.get("top_match") or "EPO OPS authentication failed.")
        run_button = st.button("Generate checklist report", type="primary")

    if not run_button:
        st.info("Paste or upload the actual application, then generate the checklist report. Built-in .txt files are loaded as guidance, not example applications.")
        return

    progress_panel = st.empty()
    with progress_panel.container():
        status = st.status("Generating checklist report...", expanded=True)
        stage_display = st.empty()
        progress = st.progress(0)

    try:
        generation = run_generation_pipeline(
            app_text=app_text,
            app_uploads=app_uploads,
            call_text=call_text,
            call_uploads=call_uploads,
            settings=settings,
            run_similarity=run_similarity,
            mock_similarity=mock_similarity,
            stage_display=stage_display,
            progress=progress,
        )
    except Exception as exc:
        _set_generation_stage(stage_display, f"Generation stopped at this stage: {exc}")
        status.update(label="Checklist report generation failed", state="error", expanded=True)
        st.error(f"Checklist report could not be generated. {exc}")
        return

    facts = generation["facts"]
    specific_reqs = generation["specific_reqs"]
    baseline = generation["baseline"]
    checklist = generation["checklist"]
    dashboard = generation["dashboard"]
    priority = generation["priority"]
    similarity = generation["similarity"]

    _advance_generation_progress(progress, 96, 100, GENERATION_PROGRESS_STEP_DELAY_SECONDS)
    status.update(label="Checklist report generated", state="complete", expanded=False)
    sleep(0.4)
    progress_panel.empty()

    tab_summary, tab_checklist, tab_rag, tab_similarity, tab_priority, tab_raw = st.tabs([
        "Summary",
        "Checklist Report",
        "RAG Dashboard",
        "Similarity Check",
        "Priority Missing Evidence",
        "Raw JSON",
    ])

    with tab_summary:
        if not specific_reqs:
            st.info(NO_SPECIFIC_CALL_GUIDANCE_MESSAGE)
        st.markdown(render_main_case_summary(facts, dashboard, priority))
        st.markdown(render_executive_review_note(facts, dashboard, priority))
    with tab_checklist:
        st.markdown(render_checklist_report_summary(checklist, facts))
        st.markdown("## Detailed checklist table")
        st.dataframe(checklist_table_rows(checklist), use_container_width=True)
        if show_raw_requirements:
            st.subheader("Developer: raw extracted requirements")
            st.json([req.__dict__ for req in baseline + specific_reqs])
    with tab_rag:
        st.markdown(render_rag_dashboard_summary(dashboard))
        st.markdown("## Detailed RAG dashboard")
        st.dataframe(dashboard_table_rows(dashboard), use_container_width=True)
        warnings = [w for row in dashboard for w in row.get("hard_validation_warnings", [])]
        if warnings:
            st.warning("; ".join(warnings))
    with tab_similarity:
        st.markdown(render_similarity_check_summary(similarity))
        st.markdown("## Detailed similarity results")
        st.write("Similarity uses live APIs only when explicitly enabled and privacy gates allow it. Normal flow does not simulate results.")
        st.dataframe(similarity_table_rows(similarity["results"]), use_container_width=True)
        st.caption("Query terms: " + similarity_query_terms_display(similarity["query"].primary_terms + similarity["query"].secondary_terms))
    with tab_priority:
        st.markdown(priority)
    with tab_raw:
        st.markdown(render_raw_json_note())
        st.json(raw_json_payload(facts=facts, checklist=checklist, dashboard=dashboard, similarity=similarity))


if __name__ == "__main__":
    main()
