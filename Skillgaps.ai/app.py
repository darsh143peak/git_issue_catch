import asyncio
import os
import sys

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ── STABLE WINDOWS COROUTINE OVERRIDE POLICY ──
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from github_matcher import GitHubMCPIssueMatcher

load_dotenv()

st.set_page_config(page_title="GitMatch.AI Dashboard", layout="wide")

@st.cache_resource(show_spinner="Loading semantic matcher...")
def get_matcher() -> GitHubMCPIssueMatcher:
    return GitHubMCPIssueMatcher()


def run_async(coro):
    return asyncio.run(coro)


def normalize_repo_input(owner: str, repo: str) -> tuple[str, str]:
    owner = owner.strip().strip("/")
    repo = repo.strip().strip("/")
    if "/" in owner and not repo:
        owner, repo = owner.split("/", 1)
    if "/" in repo and not owner:
        owner, repo = repo.split("/", 1)
    return owner.strip(), repo.strip()

matcher = get_matcher()

st.title("GitMatch.AI")
st.subheader("Automated developer-to-open-source alignment powered by MCP")

with st.sidebar:
    st.header("MCP Authorization")
    existing_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    git_token = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        value=existing_token,
        help="Loaded from .env as GITHUB_PERSONAL_ACCESS_TOKEN. Sidebar input overrides it for this session.",
    )
    if git_token:
        os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = git_token.strip()
        matcher.token = git_token.strip()

    st.divider()
    st.header("Issue Source")
    sync_mode = st.radio(
        "Data stream",
        ["Target Live GitHub Repository", "Simulated High-Impact Organizations"],
        index=0,
    )

    target_owner = "pandas-dev"
    target_repo = "pandas"
    max_issues = 20
    if sync_mode == "Target Live GitHub Repository":
        target_owner = st.text_input("Repository owner", value=target_owner).strip()
        target_repo = st.text_input("Repository name", value=target_repo).strip()
        max_issues = st.slider("Open issues to rank", min_value=5, max_value=50, value=20, step=5)

st.header("Developer Experience Blueprint")
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    domain_focus = st.selectbox(
        "Primary domain",
        ["Backend Engineering", "Machine Learning Ops", "Frontend Architecture", "Cloud Infrastructure"],
    )
    experience_level = st.select_slider(
        "Target task complexity",
        options=["Good First Issue (Junior)", "Help Wanted (Intermediate)", "Core Refactor (Senior)"],
    )

with col_input2:
    developer_skills = st.text_area(
        "Skills inventory",
        value="Python, asyncio, JSON-RPC, API parsing, pytest, Streamlit, semantic search.",
        height=130,
    )

trigger_matching = st.button("Execute Live Vector Matching", type="primary")

if trigger_matching:
    if not developer_skills.strip():
        st.error("Add at least a few skills so the semantic matcher has a profile to compare.")
        st.stop()

    with st.spinner("Fetching issues and computing semantic alignment..."):
        using_live_source = sync_mode == "Target Live GitHub Repository"
        raw_issues = []

        if using_live_source:
            target_owner, target_repo = normalize_repo_input(target_owner, target_repo)
            raw_issues = run_async(matcher.fetch_live_issues(target_owner, target_repo, max_issues=max_issues))
            if not raw_issues:
                st.warning(f"{matcher.last_error or 'Live issue fetch failed.'} Using local simulated issues for this run.")
            elif matcher.last_source == "GitHub REST fallback":
                st.caption("Live issues loaded through GitHub REST fallback.")

        if not raw_issues:
            raw_issues = matcher._generate_dynamic_fallback_issues(domain_focus, experience_level)

        # Filter issues by the selected experience level using labels/difficulty inference
        filtered_issues = matcher.filter_issues_by_experience(raw_issues, experience_level)
        if not filtered_issues:
            st.warning(f"No issues matched the difficulty '{experience_level}'. Showing unfiltered results.")
            filtered_issues = raw_issues
        raw_issues = filtered_issues

        ranked_df = matcher.compute_semantic_matches(developer_skills, raw_issues, experience_level)

    if ranked_df.empty:
        st.error("No matchable issues were generated. Check the repository name, token, and skills input.")
        st.stop()

    st.header("Recommendation Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Open tasks indexed", len(ranked_df))
    m2.metric("Top match", f"{ranked_df['match_score'].max() * 100:.2f}%")
    m3.metric("Embedding model", matcher.model_name)

    c_left, c_right = st.columns([2, 3])

    with c_left:
        st.subheader("Semantic Alignment Scores")
        chart_df = ranked_df.copy()
        chart_df["issue"] = chart_df["repo"].astype(str) + " #" + chart_df["id"].astype(str)
        fig = px.bar(
            chart_df,
            x="issue",
            y="match_score",
            hover_data=["repo", "title", "labels"],
            hover_name="difficulty",
            labels={"issue": "Issue", "match_score": "Match score"},
            color="match_score",
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.subheader("Top-Ranked Contributions")
        for _, row in ranked_df.iterrows():
            score_pct = row["match_score"] * 100
            with st.container(border=True):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.markdown(f"### `{row['repo']}` issue #{row['id']}")
                    st.markdown(f"**Title:** {row['title']}")
                    st.markdown(f"**Difficulty:** `{row['difficulty']}`")
                    st.markdown(f"**Labels:** `{row['labels']}`")
                    # show a short excerpt of the body to help judge suitability
                    if 'body' in row and row['body']:
                        excerpt = (row['body'][:300] + '...') if len(row['body']) > 300 else row['body']
                        st.markdown(f"**Excerpt:** {excerpt}")
                with col_b:
                    st.markdown(f"#### {score_pct:.1f}%")
                    st.link_button("View", row["url"])

    with st.expander("Raw ranked data"):
        st.dataframe(
            ranked_df[["repo", "id", "title", "difficulty", "labels", "semantic_score", "difficulty_score", "match_score", "url"]],
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption("GitMatch.AI uses GitHub MCP over stdio when available, then falls back gracefully for offline demos.")
