import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="IndiaRuns — Candidate Ranking Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

B1_COLOR = "#636efa"
B2_COLOR = "#ef553b"
B3_COLOR = "#00cc96"
B4_COLOR = "#ab63fa"
B5_COLOR = "#ffa15a"
COMPOSITE_COLOR = "#19d3f3"

SCORE_META = {
    "semantic_score": {"label": "B1 Semantic", "color": B1_COLOR, "weight": 0.35},
    "trajectory_score": {"label": "B2 Trajectory", "color": B2_COLOR, "weight": 0.25},
    "stability_score": {"label": "B3 Stability", "color": B3_COLOR, "weight": 0.15},
    "platform_score": {"label": "B4 Platform", "color": B4_COLOR, "weight": 0.20},
    "cert_bonus": {"label": "B5 Cert Bonus", "color": B5_COLOR, "weight": 0.05},
}

PRESETS = {
    "Senior Data Engineer": "data/outputs/de",
    "HR Manager": "data/outputs/hr",
    "Mid Data Analyst": "data/outputs/analyst",
    "Project Manager": "data/outputs/pm",
    "Senior Accountant": "data/outputs/accountant",
}

AVAILABLE_PRESETS: dict[str, str] = {}
for name, path in PRESETS.items():
    if Path(path, "ranked_output.csv").exists():
        AVAILABLE_PRESETS[name] = path


def make_score_bar(fig, results_df: pd.DataFrame) -> go.Figure:
    score_cols = [k for k in SCORE_META]
    melted = results_df.melt(
        id_vars=["rank", "candidate_id", "composite_score"],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score",
    )
    melted["score_label"] = melted["score_type"].map(lambda x: SCORE_META[x]["label"])
    melted["color"] = melted["score_type"].map(lambda x: SCORE_META[x]["color"])

    fig = px.bar(
        melted,
        x="candidate_id",
        y="score",
        color="score_label",
        color_discrete_map={v["label"]: v["color"] for v in SCORE_META.values()},
        barmode="group",
        title="Per-Score Breakdown by Candidate",
        labels={"candidate_id": "Candidate", "score": "Score", "score_label": "Score Type"},
        height=400,
    )
    fig.add_scatter(
        x=results_df["candidate_id"],
        y=results_df["composite_score"],
        mode="lines+markers",
        name="Composite",
        line=dict(color=COMPOSITE_COLOR, width=3, dash="dot"),
        marker=dict(size=10, symbol="diamond"),
        yaxis="y",
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def make_radar_chart(row: pd.Series) -> go.Figure:
    categories = [v["label"] for v in SCORE_META.values()]
    values = [row[k] for k in SCORE_META]
    values += values[:1]
    theta = categories + categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill="toself",
        name=f"{row['candidate_id']} ({row['composite_score']:.3f})",
        line=dict(color=COMPOSITE_COLOR, width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=350,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def make_weighted_composition_chart() -> go.Figure:
    labels = [v["label"] for v in SCORE_META.values()]
    weights = [v["weight"] for v in SCORE_META.values()]
    colors = [v["color"] for v in SCORE_META.values()]
    fig = go.Figure(data=[go.Pie(labels=labels, values=weights, marker=dict(colors=colors))])
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def candidate_detail_view(row: pd.Series) -> None:
    cid = row["candidate_id"]
    st.subheader(f"Candidate: {cid}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Composite", f"{row['composite_score']:.3f}")
    col2.metric("Semantic", f"{row['semantic_score']:.3f}")
    col3.metric("Trajectory", f"{row['trajectory_score']:.3f}")
    col4.metric("Stability", f"{row['stability_score']:.3f}")
    col5.metric("Platform", f"{row['platform_score']:.3f}")
    col6.metric("Cert Bonus", f"{row['cert_bonus']:.3f}")

    col_radar, col_pie = st.columns([3, 2])
    with col_radar:
        st.plotly_chart(make_radar_chart(row), use_container_width=True, key=f"candidate_radar_{row['candidate_id']}")
    with col_pie:
        st.plotly_chart(make_weighted_composition_chart(), use_container_width=True, key=f"candidate_pie_{row['candidate_id']}")

    with st.expander("Match Summary", expanded=True):
        st.markdown(row.get("match_summary", "*No summary*"))
    with st.expander("Skill Alignment"):
        st.markdown(row.get("skill_alignment", "*No data*"))
    with st.expander("Seniority Assessment"):
        st.markdown(row.get("seniority_assessment", "*No data*"))
    with st.expander("Trajectory Signal"):
        st.markdown(row.get("trajectory_signal", "*No data*"))
    with st.expander("Platform Summary"):
        st.markdown(row.get("platform_summary", "*No data*"))
    with st.expander("Flags"):
        st.markdown(row.get("flags", "*No flags*"))

    grounding = row.get("grounding_validated", False)
    if grounding:
        st.success("✅ Grounding validated — explanation cites real evidence")
    else:
        st.warning("⚠️ Grounding NOT validated — LLM may have hallucinated")


def load_precomputed(path: str) -> pd.DataFrame:
    csv_path = Path(path, "ranked_output.csv")
    df = pd.read_csv(csv_path)
    if "grounding_validated" in df.columns:
        df["grounding_validated"] = df["grounding_validated"].astype(bool)
    return df


def show_results(df: pd.DataFrame) -> None:
    st.divider()
    st.header("📊 Results Dashboard")

    grounding_count = df["grounding_validated"].sum() if "grounding_validated" in df.columns else 0
    st.metric(
        "Grounding Validation",
        f"{grounding_count}/20",
        help="Number of candidates with citation-backed LLM explanations",
    )

    tab_leaderboard, tab_detail, tab_charts, tab_export = st.tabs(
        ["🏆 Leaderboard", "🔍 Candidate Detail", "📈 Charts & Analysis", "📥 Export"]
    )

    with tab_leaderboard:
        display_cols = ["rank", "candidate_id", "composite_score",
                        "semantic_score", "trajectory_score", "stability_score",
                        "platform_score", "cert_bonus", "grounding_validated"]
        display_df = df[display_cols].copy()
        display_df.columns = ["Rank", "Candidate ID", "Composite", "B1 Semantic",
                               "B2 Trajectory", "B3 Stability", "B4 Platform",
                               "B5 Cert", "Grounded"]

        col_config = {}
        for c in ["Composite", "B1 Semantic", "B2 Trajectory", "B3 Stability", "B4 Platform", "B5 Cert"]:
            col_config[c] = st.column_config.ProgressColumn(
                c, format="%.3f", min_value=0, max_value=1,
            )

        st.dataframe(
            display_df,
            column_config=col_config,
            use_container_width=True,
            hide_index=True,
            height=560,
        )

        with st.expander("Raw explanations for top-20"):
            for _, row in df.iterrows():
                st.markdown(f"**#{row['rank']} — {row['candidate_id']}** (Composite: {row['composite_score']:.3f})")
                summary = row.get("match_summary", "*No summary*")
                st.markdown(f"> {summary[:200]}...")
                st.markdown("---")

    with tab_detail:
        selected_rank = st.selectbox(
            "Select candidate by rank",
            options=sorted(df["rank"].tolist()),
            format_func=lambda r: f"#{r} — {df[df['rank']==r]['candidate_id'].values[0]} ({df[df['rank']==r]['composite_score'].values[0]:.3f})",
        )
        row = df[df["rank"] == selected_rank].iloc[0]
        candidate_detail_view(row)

    with tab_charts:
        st.subheader("Score Breakdown by Candidate")
        fig_bars = make_score_bar(None, df)
        st.plotly_chart(fig_bars, use_container_width=True, key="score_breakdown_bars")

        st.subheader("Score Distribution")
        score_cols = [k for k in SCORE_META] + ["composite_score"]
        score_labels = [SCORE_META[k]["label"] for k in SCORE_META] + ["Composite"]
        colors = [SCORE_META[k]["color"] for k in SCORE_META] + [COMPOSITE_COLOR]
        fig_dist = go.Figure()
        for col, label, color in zip(score_cols, score_labels, colors):
            fig_dist.add_trace(go.Box(
                y=df[col], name=label, marker_color=color, boxmean="sd",
            ))
        fig_dist.update_layout(height=400, yaxis=dict(range=[0, 1]),
                               title="Score Distributions Across Top-20")
        st.plotly_chart(fig_dist, use_container_width=True, key="score_distribution_box")

        st.subheader("Radar Comparison (Top 5)")
        top5 = df.head(5)
        fig_radar = go.Figure()
        categories = [v["label"] for v in SCORE_META.values()]
        for _, row in top5.iterrows():
            values = [row[k] for k in SCORE_META]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                name=f"{row['candidate_id']} ({row['composite_score']:.3f})",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_comparison")

        st.subheader("Weight Composition")
        st.plotly_chart(make_weighted_composition_chart(), use_container_width=True, key="weighted_composition")

    with tab_export:
        st.download_button(
            "📥 Download as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"indiaruns_output_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        st.download_button(
            "📥 Download as JSON",
            data=df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name=f"indiaruns_output_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


def main() -> None:
    st.title("IndiaRuns — Candidate Ranking Engine")
    st.markdown("AI-powered candidate-to-JD matching with multi-faceted scoring and LLM-grounded explanations.")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    has_groq = bool(groq_key)
    has_presets = len(AVAILABLE_PRESETS) > 0

    st.sidebar.markdown("### Mode")
    if has_presets and not has_groq:
        st.sidebar.success("📁 Demo Mode — browsing pre-computed results")
        st.sidebar.caption("Set GROQ_API_KEY for live pipeline mode")
    elif has_presets and has_groq:
        st.sidebar.info("🔬 Live Mode — pipeline will run on Submit")
        st.sidebar.caption("Pre-computed results also available")
    elif not has_presets:
        st.sidebar.warning("⚠️ No pre-computed results found")
        if not has_groq:
            st.sidebar.error("Set GROQ_API_KEY and run index first")

    # ── Demo mode: select pre-computed JD ──
    if has_presets and not has_groq:
        selected_jd = st.sidebar.selectbox(
            "Select Job Description",
            options=list(AVAILABLE_PRESETS.keys()),
            index=0,
        )
        preset_path = AVAILABLE_PRESETS[selected_jd]
        df = load_precomputed(preset_path)

        st.sidebar.divider()
        st.sidebar.markdown("### Quick Start")
        st.sidebar.markdown(
            "1. Select a JD from the dropdown\n"
            "2. Browse the leaderboard & candidate details\n"
            "3. Export results as CSV or JSON"
        )

        show_results(df)
        st.sidebar.divider()
        st.sidebar.caption("Built with Streamlit + SentenceTransformers + FAISS + Groq")
        return

    # ── Live mode (GROQ key present) ──
    if has_groq:
        try:
            from src.config import Settings
            from src.llm import create_llm_client, LLMClient
            from src.jd_parser.parser import parse_job_description
            from src.pipeline.query_pipeline import run_query
            from src.models import RankedResult
        except ImportError as e:
            st.error(f"Source modules not available in this deployment: {e}")
            st.info("Fallback to demo mode (if available)")
            if has_presets:
                selected_jd = st.sidebar.selectbox(
                    "Select Job Description",
                    options=list(AVAILABLE_PRESETS.keys()),
                    index=0,
                )
                df = load_precomputed(AVAILABLE_PRESETS[selected_jd])
                show_results(df)
            return

        settings = Settings.from_yaml(Path("config/settings.yaml"))

        processed_dir = st.sidebar.text_input(
            "Processed Index Directory",
            value=settings.paths.processed_dir,
        )
        output_dir = st.sidebar.text_input(
            "Output Directory",
            value="data/outputs/demo",
        )

        index_ready = Path(processed_dir, "candidate_index.faiss").exists()
        if not index_ready:
            st.sidebar.warning("⚠️ Index not found — run `python main.py index` first")

        st.sidebar.divider()
        st.sidebar.markdown("### Model Configuration")
        st.sidebar.info(
            f"**Embedding:** {settings.embedding.model_name}\n\n"
            f"**LLM:** {settings.llm.model} ({settings.llm.provider})\n\n"
            f"**Weights:** B1={settings.scoring_weights.semantic}, B2={settings.scoring_weights.trajectory}, "
            f"B3={settings.scoring_weights.stability}, B4={settings.scoring_weights.platform}, "
            f"B5={settings.scoring_weights.cert_bonus_multiplier}"
        )

        st.sidebar.divider()
        st.sidebar.markdown("### Quick Start")
        st.sidebar.markdown(
            "1. Paste a JD in the text area\n"
            "2. Click **Run Pipeline**\n"
            "3. Explore the leaderboard & candidate details"
        )

        jd_input = st.text_area(
            "📝 Paste the Job Description",
            height=250,
            placeholder="Paste the full job description here...",
        )
        uploaded_file = st.file_uploader("Or upload a JD file", type=["txt", "md", "pdf"])

        run_disabled = False
        if not index_ready:
            run_disabled = True
        elif not jd_input and not uploaded_file:
            run_disabled = True

        col_run, col_status = st.columns([1, 3])
        with col_run:
            run_clicked = st.button("🚀 Run Pipeline", disabled=run_disabled, type="primary")

        if "results" not in st.session_state:
            st.session_state.results = None
        if "timing" not in st.session_state:
            st.session_state.timing = {}
        if "df" not in st.session_state:
            st.session_state.df = None

        if run_clicked and (jd_input or uploaded_file):
            if uploaded_file:
                jd_text = uploaded_file.read().decode("utf-8")
            else:
                jd_text = jd_input

            with st.spinner("Running pipeline (parsing JD, retrieving, scoring, explaining)..."):
                t_start = time.time()
                try:
                    llm_client = create_llm_client(settings)
                    results = run_query(jd_text, Path(processed_dir), Path(output_dir), settings, llm_client)
                    elapsed = time.time() - t_start
                    st.session_state.results = results
                    st.session_state.timing["total"] = elapsed

                    rows = []
                    for r in results:
                        rows.append({
                            "rank": r.rank,
                            "candidate_id": r.candidate_id,
                            "composite_score": r.composite_score,
                            "semantic_score": r.semantic_score,
                            "trajectory_score": r.trajectory_score,
                            "stability_score": r.stability_score,
                            "platform_score": r.platform_score,
                            "cert_bonus": r.cert_bonus,
                            "match_summary": r.explanation.match_summary,
                            "skill_alignment": r.explanation.skill_alignment,
                            "seniority_assessment": r.explanation.seniority_assessment,
                            "trajectory_signal": r.explanation.trajectory_signal,
                            "platform_summary": r.explanation.platform_summary,
                            "flags": r.explanation.flags,
                            "grounding_validated": r.explanation.grounding_validated,
                        })
                    df = pd.DataFrame(rows)
                    st.session_state.df = df

                    st.success(f"Pipeline completed in {elapsed:.1f}s — top 20 results ready!")
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if st.session_state.df is not None:
            show_results(st.session_state.df)
            st.sidebar.divider()
            st.sidebar.markdown("### Pipeline Timing")
            total = st.session_state.timing.get("total", 0)
            st.sidebar.metric("Total Pipeline", f"{total:.1f}s")
        else:
            st.info("👈 Paste a JD and click **Run Pipeline** to see results")
    else:
        st.error("No GROQ_API_KEY set and no pre-computed results available.")
        st.info("Set GROQ_API_KEY as an environment variable or deploy pre-computed CSVs to `data/outputs/`.")


if __name__ == "__main__":
    main()
