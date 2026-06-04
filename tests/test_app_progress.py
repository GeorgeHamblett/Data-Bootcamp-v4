from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import app
from settings import Settings


APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


class FakeStageDisplay:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def markdown(self, message: str) -> None:
        self.messages.append(message)


class FakeProgress:
    def __init__(self) -> None:
        self.values: list[int] = []

    def progress(self, value: int) -> None:
        self.values.append(value)


def test_app_uses_visible_status_and_progress_workflow() -> None:
    assert 'st.status("Generating checklist report...", expanded=True)' in APP_SOURCE
    assert "st.progress(0)" in APP_SOURCE
    assert "progress_panel = st.empty()" in APP_SOURCE
    assert "progress_panel.empty()" in APP_SOURCE
    assert "stage_display = st.empty()" in APP_SOURCE
    assert "_set_generation_stage" in APP_SOURCE
    assert "_update_generation_progress" in APP_SOURCE
    assert "status.write" not in APP_SOURCE
    assert "Checklist report generated" in APP_SOURCE
    assert "Checklist report generation failed" in APP_SOURCE
    assert "st.spinner" not in APP_SOURCE

    for stage in [
        "1. Reading application documents",
        "2. Extracting application facts",
        "3. Reading optional funding call guidance",
        "4. Selecting relevant guidance",
        "5. Building the checklist review",
        "6. Building the RAG dashboard",
        "7. Prioritising missing evidence",
        "8. Running similarity/novelty check",
        "9. Preparing report tabs",
    ]:
        assert stage in APP_SOURCE


def test_generation_still_defines_the_same_six_report_tabs() -> None:
    tree = ast.parse(APP_SOURCE)
    tab_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tabs"
    ]
    assert len(tab_calls) == 1
    tab_names = [elt.value for elt in tab_calls[0].args[0].elts]
    assert tab_names == [
        "Summary",
        "Checklist Report",
        "RAG Dashboard",
        "Similarity Check",
        "Priority Missing Evidence",
        "Raw JSON",
    ]


def test_generation_pipeline_calls_main_functions_in_order(monkeypatch) -> None:
    calls: list[str] = []

    def fake_combine(pasted, uploads):
        calls.append("combine_application" if pasted == "application" else "combine_guidance")
        return [SimpleNamespace(name="doc", text=f"{pasted} document text")]

    def fake_extract(docs):
        calls.append("extract_application_facts")
        return SimpleNamespace(
            application_claimed_call="PDA",
            product_or_intervention="product",
            technology_type="device",
            trl_evidence="TRL 4",
        )

    def fake_detects(*args, **kwargs):
        calls.append("detects_pda_relevance")
        return True

    def fake_baseline(*args, **kwargs):
        calls.append("build_baseline_requirement_bank")
        return [SimpleNamespace(overrides_general_guidance=False)]

    def fake_parse(*args, **kwargs):
        calls.append("parse_guidance_text")
        return [SimpleNamespace(overrides_general_guidance=False)]

    def fake_checklist(*args, **kwargs):
        calls.append("build_checklist")
        return ["checklist"]

    def fake_dashboard(*args, **kwargs):
        calls.append("build_rag_dashboard")
        return ["dashboard"]

    def fake_priority(*args, **kwargs):
        calls.append("render_priority_missing_evidence")
        return "priority"

    def fake_similarity(*args, **kwargs):
        calls.append("run_similarity_service")
        assert kwargs["run_similarity_check"] is False
        assert kwargs["mock_mode"] is False
        return {"query": SimpleNamespace(primary_terms=[], secondary_terms=[]), "results": []}

    monkeypatch.setattr(app, "combine_pasted_and_uploaded", fake_combine)
    monkeypatch.setattr(app, "extract_application_facts", fake_extract)
    monkeypatch.setattr(app, "detects_pda_relevance", fake_detects)
    monkeypatch.setattr(app, "build_baseline_requirement_bank", fake_baseline)
    monkeypatch.setattr(app, "parse_guidance_text", fake_parse)
    monkeypatch.setattr(app, "build_checklist", fake_checklist)
    monkeypatch.setattr(app, "build_rag_dashboard", fake_dashboard)
    monkeypatch.setattr(app, "render_priority_missing_evidence", fake_priority)
    monkeypatch.setattr(app, "run_similarity_service", fake_similarity)

    stage_display = FakeStageDisplay()
    progress = FakeProgress()
    result = app.run_generation_pipeline(
        app_text="application",
        app_uploads=[],
        call_text="guidance",
        call_uploads=[],
        settings=Settings(),
        run_similarity=False,
        mock_similarity=False,
        stage_display=stage_display,
        progress=progress,
        progress_delay_seconds=0,
    )

    assert calls == [
        "combine_application",
        "extract_application_facts",
        "combine_guidance",
        "detects_pda_relevance",
        "build_baseline_requirement_bank",
        "parse_guidance_text",
        "build_checklist",
        "build_rag_dashboard",
        "render_priority_missing_evidence",
        "run_similarity_service",
    ]
    assert result["priority"] == "priority"
    assert progress.values == [8, 12, 20, 32, 44, 58, 70, 80, 88, 96]
    assert stage_display.messages[-1].startswith("**9. Preparing report tabs")
    assert all(message.startswith("**") for message in stage_display.messages)


def test_progress_can_advance_gradually_with_a_delay(monkeypatch) -> None:
    sleeps: list[float] = []
    progress = FakeProgress()

    monkeypatch.setattr(app, "sleep", sleeps.append)

    result = app._advance_generation_progress(progress, 0, 3, 0.08)

    assert result == 3
    assert progress.values == [1, 2, 3]
    assert sleeps == [0.08, 0.08, 0.08]


def test_similarity_checking_remains_optional_and_privacy_gated() -> None:
    assert app._similarity_progress_message(False, False, Settings()) == (
        "Similarity/novelty check skipped because it was not enabled."
    )
    assert "mock mode" in app._similarity_progress_message(True, True, Settings())
    assert "blocked by strict local-only mode" in app._similarity_progress_message(
        True,
        False,
        Settings(strict_local_only_mode=True),
    )
    assert "blocked by privacy settings" in app._similarity_progress_message(
        True,
        False,
        Settings(allow_external_similarity_queries=False),
    )
    assert "safe-query-term privacy gate" in app._similarity_progress_message(
        True,
        False,
        Settings(send_only_safe_query_terms=False),
    )
    live_message = app._similarity_progress_message(True, False, Settings())
    assert "cleaned concept terms only" in live_message
    assert "no full application text" in live_message


def test_progress_messages_do_not_claim_gdpr_or_display_secret_values() -> None:
    lower_source = APP_SOURCE.lower()
    assert "gdpr compliant" not in lower_source
    assert "api key" not in "\n".join(
        line.lower() for line in APP_SOURCE.splitlines() if "_set_generation_stage" in line or "_update_generation_progress" in line
    )
