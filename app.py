# ── requirements ──────────────────────────────────────────────────────────────
# pip install streamlit plotly opencv-python ultralytics
# pip install requests            (Ollama backend)
# pip install openai              (OpenAI backend)
# pip install scikit-learn scipy  (tactical_analyzer package)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Football Tactical Analyzer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* formation badge */
  .formation-badge {
    display: inline-block;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 0.3rem 1.2rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 2px solid rgba(255,255,255,0.18);
    color: #fff;
    text-shadow: 0 1px 6px rgba(0,0,0,0.5);
    margin-bottom: 0.6rem;
  }
  /* team section header */
  .team-header {
    font-size: 1.1rem;
    font-weight: 700;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }
  .team-a { background: rgba(76,155,232,0.20); border-left: 4px solid #4C9BE8; }
  .team-b { background: rgba(232,90,76,0.20);  border-left: 4px solid #E85A4C; }
  /* label pill */
  .label-pill {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.65rem;
    border-radius: 20px;
    margin: 0.15rem 0.2rem 0.15rem 0;
    background: rgba(255,255,255,0.1);
    color: #ddd;
    border: 1px solid rgba(255,255,255,0.15);
  }
  /* placeholder card */
  .placeholder-card {
    height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.4);
    font-size: 0.9rem;
  }
  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
</style>
""", unsafe_allow_html=True)

# ── team display config ───────────────────────────────────────────────────────
_TEAM_COLORS  = {1: "#4C9BE8", 2: "#E85A4C"}
_TEAM_LABELS  = {1: "Team A",  2: "Team B"}
_TEAM_CSS     = {1: "team-a",  2: "team-b"}

# ── Backend imports ───────────────────────────────────────────────────────────
from api.pipeline_runner import execute_pipeline
from api.result_adapter import normalize_evaluation
from tactical_analyzer import TacticalNarrator

def _eval_tactics(analysis_result, possession_pct, total_distance):
    """Derive tactical labels from pipeline time-series samples."""
    result = {}
    for team_id, samples in analysis_result.items():
        compacts  = [s["compact"]   for s in samples if s.get("compact")   is not None]
        pressings = [s["pressing"]  for s in samples if s.get("pressing")  is not None]
        speeds    = [s["avg_speed"] for s in samples if s.get("avg_speed") is not None]

        avg_c = float(np.mean(compacts))  if compacts  else None
        avg_p = float(np.mean(pressings)) if pressings else None

        result[team_id] = {
            "formation":         Counter(s["formation"] for s in samples
                                         if s["formation"] != "unknown").most_common(1)[0][0]
                                 if any(s["formation"] != "unknown" for s in samples) else "unknown",
            "compactness_label": ("very_compact" if avg_c and avg_c < 15
                                  else "compact" if avg_c and avg_c < 22
                                  else "stretched" if avg_c and avg_c < 30
                                  else "disorganized") if avg_c else "unknown",
            "pressing_label":    ("high_press" if avg_p and avg_p > 0.6
                                  else "mid_block" if avg_p and avg_p > 0.3
                                  else "low_block") if avg_p else "unknown",
            "speed_trend":       "consistent_intensity",
            "possession_label":  ("dominant_possession"
                                  if (possession_pct.get(team_id) or 0) > 60
                                  else "balanced"
                                  if (possession_pct.get(team_id) or 0) > 40
                                  else "under_pressure"),
            "flags":             [],
        }
    result["match_events"] = []
    return result


def _format_report_markdown(evaluation: dict) -> str:
    lines: list[str] = []
    overview = evaluation.get("tong_quan_tran_dau") or evaluation.get("overview") or {}
    if overview.get("nhan_xet_chung"):
        lines.append("## Tổng quan\n")
        lines.append(overview["nhan_xet_chung"] + "\n")

    for key, label in [("doi_1", "Đội 1"), ("doi_2", "Đội 2")]:
        team = evaluation.get(key) or evaluation.get("danh_gia_doi", {}).get(key, {})
        if not team:
            continue
        lines.append(f"## {label}\n")
        lines.append(
            f"**Điểm:** {team.get('diem_so_tong', team.get('diem_tong', '—'))} "
            f"({team.get('xep_loai', '—')})\n"
        )
        if team.get("chien_thuat"):
            lines.append(team["chien_thuat"] + "\n")
        for s in team.get("diem_manh", []):
            lines.append(f"- ✅ {s}")
        for w in team.get("diem_yeu", []):
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    if evaluation.get("ket_luan"):
        lines.append("## Kết luận\n")
        lines.append(evaluation["ket_luan"])
    return "\n".join(lines) if lines else "Không có dữ liệu báo cáo."


def _gen_report(evaluation, llm_provider=None, match_report=None):
    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY chưa được cấu hình.")
        if not match_report:
            raise ValueError("match_report không có — chạy pipeline trước.")
        narrator = TacticalNarrator(api_key=api_key)
        llm_out = narrator.analyze(match_report)
        evaluation = normalize_evaluation(llm_out, match_report)
        return _format_report_markdown(evaluation)

    if llm_provider == "ollama":
        if not match_report:
            raise ValueError("match_report không có — chạy pipeline trước.")
        narrator = TacticalNarrator(
            api_key="ollama",
            model=os.getenv("LLM_MODEL", "llama3"),
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        )
        llm_out = narrator.analyze(match_report)
        evaluation = normalize_evaluation(llm_out, match_report)
        return _format_report_markdown(evaluation)

    return _format_report_markdown(evaluation or {})


def run_pipeline(video_path: str) -> dict:
    """Run the full CV + tactical pipeline on *video_path*."""
    outputs = execute_pipeline(video_path, read_from_stub=False)
    payload = outputs["streamlit"]
    payload["match_report"] = outputs["match_report"]
    payload["evaluation"] = outputs["adapted"]["evaluation"]
    return payload


# ── chart helpers ─────────────────────────────────────────────────────────────

def _make_timeseries_chart(
    analysis_result: dict,
    metric_key: str,
    y_label: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for team_id, samples in sorted(analysis_result.items()):
        xs = [s["frame"] for s in samples]
        ys = [s.get(metric_key) for s in samples]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=_TEAM_LABELS.get(team_id, f"Team {team_id}"),
            line=dict(color=_TEAM_COLORS.get(team_id, "#aaa"), width=2),
            marker=dict(size=5),
            connectgaps=True,
        ))
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Frame",
        yaxis_title=y_label,
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _formation_timeline_df(analysis_result: dict) -> pd.DataFrame:
    """Build a filtered dataframe showing only frames where a formation changes."""
    all_frames = sorted({s["frame"] for samples in analysis_result.values() for s in samples})
    frame_map: dict = {}
    for team_id, samples in analysis_result.items():
        for s in samples:
            frame_map.setdefault(s["frame"], {})[team_id] = s["formation"]

    rows = []
    prev: dict = {}
    for f in all_frames:
        row_data = frame_map.get(f, {})
        if any(row_data.get(tid) != prev.get(tid) for tid in analysis_result):
            row = {"Frame": f}
            for tid in sorted(analysis_result.keys()):
                row[_TEAM_LABELS.get(tid, f"Team {tid}")] = row_data.get(tid, "—")
            rows.append(row)
            prev = dict(row_data)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _placeholder(msg: str = "No data yet") -> None:
    st.markdown(
        f'<div class="placeholder-card">⬚ {msg}</div>',
        unsafe_allow_html=True,
    )


# ── flag → alert mapping ──────────────────────────────────────────────────────

_FLAG_ALERTS: dict[str, tuple[str, str]] = {
    "fatigue_detected":        ("warning", "⚡ **Fatigue detected** — speed dropped >15% in the second half"),
    "tactical_shift_detected": ("warning", "🔄 **Tactical shift** — formation changed more than 3 times"),
    "under_pressure":          ("error",   "🔴 **Under pressure** — possession below 40%"),
    "dominant_possession":     ("success", "🟢 **Dominant possession** — ball control above 60%"),
    "high_intensity_running":  ("info",    "🏃 **High intensity** — total distance above 100 km"),
}

_PRESSING_ALERTS: dict[str, tuple[str, str]] = {
    "high_press": ("success", "⚡ **High press** — sustained aggressive pressing throughout"),
    "low_block":  ("info",    "🛡️ **Deep block** — disciplined defensive shape, minimal pressing"),
}

_FUNC: dict[str, any] = {
    "warning": st.warning,
    "error":   st.error,
    "success": st.success,
    "info":    st.info,
}


def _render_flag_alerts(team_eval: dict, team_label: str) -> None:
    shown = False
    for flag in team_eval.get("flags", []):
        if flag in _FLAG_ALERTS:
            kind, msg = _FLAG_ALERTS[flag]
            _FUNC[kind](f"**{team_label}** — {msg}")
            shown = True
    press = team_eval.get("pressing_label", "")
    if press in _PRESSING_ALERTS:
        kind, msg = _PRESSING_ALERTS[press]
        _FUNC[kind](f"**{team_label}** — {msg}")
        shown = True
    if not shown:
        st.info(f"**{team_label}** — No significant tactical flags detected.")


def _label_pill(text: str) -> str:
    return f'<span class="label-pill">{text}</span>'


def _render_team_card(team_id: int, team_eval: dict, poss: float | None, dist: float | None) -> None:
    css  = _TEAM_CSS.get(team_id, "team-a")
    name = _TEAM_LABELS.get(team_id, f"Team {team_id}")
    st.markdown(f'<div class="team-header {css}">{name}</div>', unsafe_allow_html=True)

    formation = team_eval.get("formation", "—")
    st.markdown(f'<div class="formation-badge">{formation}</div>', unsafe_allow_html=True)

    labels_html = "".join([
        _label_pill(team_eval.get("compactness_label", "—")),
        _label_pill(team_eval.get("pressing_label",    "—")),
        _label_pill(team_eval.get("speed_trend",       "—")),
        _label_pill(team_eval.get("possession_label",  "—")),
    ])
    st.markdown(labels_html, unsafe_allow_html=True)

    flags = team_eval.get("flags", [])
    if flags:
        st.caption("🚩 " + "  ·  ".join(flags))

    col_p, col_d = st.columns(2)
    with col_p:
        st.metric("Possession", f"{poss:.1f}%" if poss is not None else "—")
    with col_d:
        st.metric("Distance covered", f"{dist/1000:.1f} km" if dist is not None else "—")


# ── session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("pipeline_result", None),
    ("evaluation",      None),
    ("report",          None),
    ("video_bytes",     None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚽ Football Analyzer")
    st.caption("Computer-vision + tactical AI pipeline")
    st.divider()

    uploaded = st.file_uploader(
        "Upload match video",
        type=["mp4", "avi"],
        help="Upload the raw match footage to analyse.",
    )

    if uploaded:
        st.video(uploaded)

    st.divider()
    llm_choice = st.radio(
        "LLM provider for reports",
        options=["None (skip)", "Ollama (local)", "OpenAI"],
        index=0,
    )
    if llm_choice == "OpenAI":
        openai_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="sk-...",
            help="Also readable from the OPENAI_API_KEY env var.",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

    llm_provider_map = {
        "None (skip)":    None,
        "Ollama (local)": "ollama",
        "OpenAI":         "openai",
    }
    selected_provider = llm_provider_map[llm_choice]

    st.divider()
    run_btn = st.button("▶  Run Analysis", use_container_width=True, type="primary")

    if run_btn:
        if not uploaded:
            st.error("Please upload a video first.")
        else:
            with st.spinner("Running pipeline… this may take a few minutes."):
                # Save uploaded bytes to a temp file
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                try:
                    result = run_pipeline(tmp_path)
                    st.session_state.pipeline_result = result
                    st.session_state.report          = None  # reset stale report

                    # Immediately run rule engine (fast, no LLM)
                    st.session_state.evaluation = _eval_tactics(
                        result["analysis_result"],
                        result["possession_pct"],
                        result["total_distance"],
                    )
                    st.success("Pipeline complete!")
                except Exception as exc:
                    st.error(f"Pipeline failed: {exc}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    if st.session_state.pipeline_result:
        st.success("✅ Results ready")
    else:
        st.info("Upload a video and click Run Analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
result     = st.session_state.pipeline_result
evaluation = st.session_state.evaluation

tab_video, tab_metrics, tab_report = st.tabs([
    "📹 Video & Tracking",
    "📊 Tactical Metrics",
    "🤖 AI Report",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video & Tracking
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.subheader("Match Video")
    col_orig, col_out = st.columns(2)

    with col_orig:
        st.caption("Uploaded footage")
        if uploaded:
            uploaded.seek(0)
            st.video(uploaded.read())
        else:
            _placeholder("No video uploaded yet")

    with col_out:
        st.caption("Annotated output")
        if result and result.get("output_video_path"):
            try:
                with open(result["output_video_path"], "rb") as vf:
                    st.video(vf.read())
            except OSError:
                _placeholder("Output video file not found")
        else:
            _placeholder("Run pipeline to generate annotated video")

    st.divider()
    st.subheader("Heatmap & Passing Network")

    col_heat, col_net = st.columns(2)

    with col_heat:
        team_toggle = st.radio(
            "Heatmap team", ["Team A", "Team B"],
            horizontal=True, key="heatmap_toggle",
        )
        if result and result.get("heatmap_path"):
            try:
                st.image(result["heatmap_path"], caption=f"Player heatmap — {team_toggle}", use_column_width=True)
            except Exception:
                _placeholder("Could not load heatmap image")
        else:
            _placeholder("Heatmap will appear after pipeline run")

    with col_net:
        st.caption("Passing network")
        if result and result.get("passing_network_path"):
            try:
                st.image(result["passing_network_path"], caption="Passing network", use_column_width=True)
            except Exception:
                _placeholder("Could not load passing network image")
        else:
            _placeholder("Passing network will appear after pipeline run")

    st.divider()
    st.subheader("Key Statistics")

    if result:
        poss = result["possession_pct"]
        dist = result["total_distance"]
        ar   = result["analysis_result"]

        all_team_ids = sorted(ar.keys())
        metric_cols  = st.columns(len(all_team_ids) * 3)

        for i, tid in enumerate(all_team_ids):
            samples    = ar[tid]
            speeds     = [s["avg_speed"] for s in samples if s.get("avg_speed") is not None]
            avg_spd    = float(np.mean(speeds)) if speeds else None
            base_col   = i * 3
            label      = _TEAM_LABELS.get(tid, f"Team {tid}")

            with metric_cols[base_col]:
                st.metric(f"⚽ {label} Possession", f"{poss.get(tid, 0):.1f}%")
            with metric_cols[base_col + 1]:
                st.metric(f"📏 {label} Distance", f"{dist.get(tid, 0)/1000:.2f} km")
            with metric_cols[base_col + 2]:
                st.metric(f"🏃 {label} Avg Speed", f"{avg_spd:.1f} km/h" if avg_spd else "—")
    else:
        st.info("Run the pipeline to see key statistics.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tactical Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    if not result or not evaluation:
        st.info("Run the pipeline to see tactical metrics.")
    else:
        ar    = result["analysis_result"]
        poss  = result["possession_pct"]
        dist  = result["total_distance"]

        # ── Team cards ────────────────────────────────────────────────────────
        st.subheader("Team Evaluation")
        team_ids  = [tid for tid in sorted(ar.keys())]
        card_cols = st.columns(len(team_ids))

        for col, tid in zip(card_cols, team_ids):
            with col:
                team_eval = evaluation.get(tid, {})
                _render_team_card(
                    tid, team_eval,
                    poss.get(tid), dist.get(tid),
                )

        st.divider()

        # ── Plotly time-series charts ──────────────────────────────────────
        st.subheader("Metrics Over Time")

        fig_compact = _make_timeseries_chart(
            ar, "compact",   "Mean pairwise distance (m)", "🔵 Compactness Score"
        )
        fig_press   = _make_timeseries_chart(
            ar, "pressing",  "Pressing intensity (0–1)",   "⚡ Pressing Intensity"
        )
        fig_speed   = _make_timeseries_chart(
            ar, "avg_speed", "Average speed (km/h)",       "🏃 Average Speed"
        )

        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(fig_compact, use_container_width=True)
        with ch2:
            st.plotly_chart(fig_press,   use_container_width=True)

        st.plotly_chart(fig_speed, use_container_width=True)

        st.divider()

        # ── Formation change timeline ──────────────────────────────────────
        st.subheader("Formation Change Timeline")
        df_timeline = _formation_timeline_df(ar)
        if df_timeline.empty:
            st.info("No formation changes detected across sampled frames.")
        else:
            st.dataframe(
                df_timeline,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Frame": st.column_config.NumberColumn("Frame", format="%d"),
                },
            )

        # ── Match events ──────────────────────────────────────────────────
        events = evaluation.get("match_events", [])
        if events:
            st.divider()
            st.subheader("Match Events")
            for ev in events:
                st.markdown(f"- {ev}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI Report
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    if not evaluation:
        st.info("Run the pipeline to generate a tactical report.")
    else:
        # ── Flag-based alerts (rule engine only, no LLM needed) ───────────
        st.subheader("Key Tactical Alerts")
        for tid in sorted(evaluation.keys()):
            if tid == "match_events":
                continue
            _render_flag_alerts(evaluation[tid], _TEAM_LABELS.get(tid, f"Team {tid}"))

        st.divider()

        # ── LLM narrative report ──────────────────────────────────────────
        st.subheader("AI Tactical Report")

        gen_col, _ = st.columns([1, 3])
        with gen_col:
            gen_btn = st.button(
                "🤖 Generate / Regenerate Report",
                disabled=(selected_provider is None),
                help="Select an LLM provider in the sidebar to enable.",
            )

        if selected_provider is None and st.session_state.report is None:
            st.info(
                "Select **Ollama (local)** or **OpenAI** in the sidebar, "
                "then click *Generate Report*."
            )

        if gen_btn and selected_provider:
            with st.spinner(f"Generating report via {selected_provider}…"):
                try:
                    st.session_state.report = _gen_report(
                        evaluation,
                        llm_provider=selected_provider,
                        match_report=result.get("match_report") if result else None,
                    )
                except ConnectionError as ce:
                    st.error(f"🔌 Connection error: {ce}")
                    st.session_state.report = None
                except EnvironmentError as ee:
                    st.error(f"🔑 Configuration error: {ee}")
                    st.session_state.report = None
                except Exception as exc:
                    st.error(f"LLM error: {exc}")
                    st.session_state.report = None

        report_text = st.session_state.report
        if report_text:
            st.markdown(report_text)
            st.divider()
            dl_col, _ = st.columns([1, 4])
            with dl_col:
                st.download_button(
                    "⬇️ Download Report (.txt)",
                    data=report_text.encode("utf-8"),
                    file_name="tactical_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        elif selected_provider and not gen_btn:
            st.caption("Click *Generate / Regenerate Report* to produce the AI narrative.")
