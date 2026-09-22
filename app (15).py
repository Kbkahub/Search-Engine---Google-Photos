"""
Google Photos Discovery Engine
AI-Powered User Research Tool for Photo Retrieval Problems
"""

import streamlit as st
import json
import re
import urllib.request
import urllib.error
from collections import Counter
from math import log

# ─────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Discovery Engine — Google Photos Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 1. CSS / THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide Streamlit chrome ── */
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
button[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapse"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.stSidebar button[kind="header"] { display: none !important; }

/* ── Force sidebar open ── */
[data-testid="stSidebar"] { min-width: 300px !important; width: 300px !important; }
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 300px !important; width: 300px !important;
    transform: none !important; display: block !important;
}

/* ── Sidebar colour (Google Blue) ── */
div[data-testid="stSidebar"] { background: #1A73E8 !important; }
div[data-testid="stSidebar"] * { color: #FFFFFF !important; }

/* ── Nav buttons ── */
div[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #FFFFFF !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    text-align: left !important;
    padding: 10px 16px !important;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.15) !important;
}

/* ── AI answer box ── */
.ai-answer {
    background: #E8F0FE; border: 1px solid #C2D7F2;
    border-left: 4px solid #1A73E8;
    border-radius: 10px; padding: 20px; margin-bottom: 20px;
}

/* ── Card style ── */
.metric-card {
    background: #FFFFFF; border-radius: 12px;
    padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    text-align: center; border: 1px solid #E0E0E0;
}
.metric-card h2 { color: #1A73E8; font-size: 2rem; margin: 0; }
.metric-card p { color: #5F6368; font-size: 0.85rem; margin: 4px 0 0 0; }

/* ── Review card ── */
.review-card {
    background: #FFFFFF; border-radius: 10px; padding: 16px 20px;
    margin-bottom: 12px; border: 1px solid #E0E0E0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.review-card .platform-tag {
    display: inline-block; background: #E8F0FE; color: #1A73E8;
    font-size: 0.72rem; padding: 2px 8px; border-radius: 4px;
    font-weight: 600; margin-right: 6px;
}
.review-card .theme-tag {
    display: inline-block; background: #F1F3F4; color: #5F6368;
    font-size: 0.7rem; padding: 2px 6px; border-radius: 4px;
    margin-right: 4px;
}
.review-card .sentiment-pos { color: #34A853; font-weight: 600; }
.review-card .sentiment-neg { color: #EA4335; font-weight: 600; }
.review-card .sentiment-mix { color: #FBBC04; font-weight: 600; }

/* ── Section header ── */
.section-hdr {
    font-size: 1.1rem; font-weight: 700; color: #202124;
    margin-top: 28px; margin-bottom: 12px;
    padding-bottom: 6px; border-bottom: 2px solid #1A73E8;
}

/* ── Opportunity card ── */
.opp-card {
    background: #FFFFFF; border-radius: 10px; padding: 20px;
    margin-bottom: 16px; border: 1px solid #E0E0E0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.opp-card h4 { color: #202124; margin: 0 0 6px 0; }
.opp-card .score { color: #1A73E8; font-weight: 700; font-size: 1.3rem; }

/* ── Progress bars ── */
.stProgress > div > div > div { background: #1A73E8 !important; }

/* ── Expander ── */
div[data-testid="stExpander"] { border: 1px solid #E0E0E0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_reviews():
    with open("data/reviews.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_survey():
    try:
        with open("data/survey.json", "r") as f:
            data = json.load(f)
            return data if data else []
    except Exception:
        return []

REVIEWS = load_reviews()
SURVEY = load_survey()

# ─────────────────────────────────────────────
# 3. TF-IDF SEARCH ENGINE
# ─────────────────────────────────────────────
class MiniSearch:
    def __init__(self, docs, field="text"):
        self.docs = docs
        self.n = len(docs)
        self.tokens = []
        self.df = Counter()
        for doc in docs:
            toks = set(re.findall(r"[a-z0-9]+", doc.get(field, "").lower()))
            self.tokens.append(toks)
            for t in toks:
                self.df[t] += 1

    def search(self, query, top_k=15):
        q_toks = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = []
        for i, doc in enumerate(self.docs):
            score = sum(
                log(self.n / self.df[t])
                for t in q_toks
                if t in self.tokens[i] and self.df[t] > 0
            )
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [self.docs[i] for _, i in scores[:top_k]]


review_engine = MiniSearch(REVIEWS, field="text")
survey_engine = MiniSearch(SURVEY, field="searchable_text") if SURVEY else None

# ─────────────────────────────────────────────
# 4. GROQ LLM
# ─────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

def call_groq(system_prompt, user_msg, api_key):
    last_error = ""
    for model in GROQ_MODELS:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 1500,
            "temperature": 0.4,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "DiscoveryEngine/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode()
            except Exception:
                pass
            last_error = f"Model {model}: HTTP {e.code} — {error_body[:200]}"
            continue
        except Exception as e:
            last_error = f"Model {model}: {e}"
            continue
    return f"__ERROR__: {last_error}"

REVIEW_SYSTEM_PROMPT = """You are a senior user researcher analyzing Google Photos user feedback about photo retrieval problems.
Answer grounded ONLY in the provided reviews. Do NOT number the sections.

Start with a direct 2-3 sentence answer — no heading, no label, just the answer.

Then a section headed **Key Themes** listing 3-5 prominent patterns with evidence.

Then a section headed **User Segments Affected** listing which types of users are most impacted.

Then a section headed **Retrieval Gap** explaining what users remember vs what the search system requires.

Quote specific reviews as evidence. Be specific, not generic."""

SURVEY_SYSTEM_PROMPT = """You are a senior user researcher analyzing a primary user survey about Google Photos usage and photo retrieval behavior.
Answer grounded ONLY in the provided survey data. Do NOT number the sections.

Start with a direct 2-3 sentence finding — no heading, no label, just the finding.

Then a section headed **Supporting Data** referencing specific responses.

Then a section headed **Demographic Patterns** noting differences by age, gender, etc.

Be specific. Reference respondent demographics when quoting."""

# ─────────────────────────────────────────────
# 5. HELPERS
# ─────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

def sentiment_class(s):
    return {"positive": "sentiment-pos", "negative": "sentiment-neg"}.get(s, "sentiment-mix")

def render_review_card(r):
    s_cls = sentiment_class(r.get("sentiment", ""))
    themes_html = "".join(
        f'<span class="theme-tag">{t.replace("_"," ")}</span>' for t in r.get("themes", [])
    )
    rating_str = f"⭐ {r['rating']}/5" if r.get("rating") else ""
    st.markdown(f"""
    <div class="review-card">
        <div style="margin-bottom:8px;">
            <span class="platform-tag">{r.get('platform','')}</span>
            {themes_html}
            <span class="{s_cls}" style="float:right;font-size:0.8rem;">{r.get('sentiment','').upper()} {rating_str}</span>
        </div>
        <div style="color:#202124;font-size:0.92rem;line-height:1.5;">"{r.get('text','')}"</div>
        <div style="color:#9AA0A6;font-size:0.75rem;margin-top:6px;">{r.get('date','')} · {r.get('user_segment','')}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. OPPORTUNITY AREAS
# ─────────────────────────────────────────────
THEME_LABELS = {
    "vague_memory_retrieval": "Vague Memory → Failed Retrieval",
    "search_accuracy": "Search Accuracy & Relevance",
    "ai_search_regression": "AI/Gemini Search Regression",
    "face_recognition": "Face Recognition Issues",
    "temporal_navigation": "Time-Based Navigation Gaps",
    "screenshot_document_search": "Screenshot & Document Retrieval",
    "location_search": "Location-Based Search Gaps",
    "organization_friction": "Organization & Structure Friction",
}

THEME_DESCRIPTIONS = {
    "vague_memory_retrieval": "Users remember visual details, emotions, or context about a photo but cannot translate these into keyword searches. The gap between human episodic memory and text-based search is the core retrieval problem.",
    "search_accuracy": "Search returns too many results, irrelevant results, or misses photos entirely. Combining criteria (person + place + time) is not supported. Users distrust search completeness.",
    "ai_search_regression": "Gemini/Ask Photos integration degraded the classic search experience. Users report slower results, AI-generated summaries instead of photos, and less accurate matches. Many disabled it.",
    "face_recognition": "Face grouping errors, splitting one person into multiple groups, merging different people, poor recognition for children aging, people of color, and group photos. Core people-search breaks down.",
    "temporal_navigation": "Incorrect timestamps from scans, downloads, or cross-device transfers. No date-range filtering in search. Scrolling fatigue in large libraries. Seasonal/relative time references not understood.",
    "screenshot_document_search": "Photos of medicines, receipts, documents, menus, and other functional captures are hard to retrieve. OCR doesn't work on older photos, handwritten text, or non-English text. No 'document type' category.",
    "location_search": "Indoor photos lack GPS data. Location services turned off means no geotagging. No location inference from nearby photos. Can't search within shared album locations.",
    "organization_friction": "Albums are incomplete, manual organization is tedious, screenshots pollute libraries, archived photos vanish from search, no custom tags, no folder structure, deleted photos resurface.",
}

def compute_opportunity_areas():
    theme_counts = Counter()
    theme_platforms = {}
    for r in REVIEWS:
        for t in r.get("themes", []):
            theme_counts[t] += 1
            theme_platforms.setdefault(t, Counter())
            theme_platforms[t][r.get("platform", "Unknown")] += 1

    max_count = max(theme_counts.values()) if theme_counts else 1
    areas = []
    for theme, count in theme_counts.most_common():
        if theme in THEME_LABELS:
            impact = round(min(10, (count / max_count) * 10), 1)
            areas.append({
                "theme": theme,
                "label": THEME_LABELS[theme],
                "description": THEME_DESCRIPTIONS.get(theme, ""),
                "impact": impact,
                "evidence_count": count,
                "platforms": dict(theme_platforms.get(theme, {})),
            })
    return areas

# ─────────────────────────────────────────────
# 7. SIDEBAR + NAVIGATION
# ─────────────────────────────────────────────
NAV_ITEMS = [
    ("dashboard", "DASHBOARD"),
    ("public_reviews", "ASK INSIGHTS FROM PUBLIC REVIEWS"),
    ("user_survey", "ASK INSIGHTS FROM USER SURVEY"),
    ("explorer", "REVIEW EXPLORER"),
    ("comparison", "COMPARISON MATRIX"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "dashboard"

with st.sidebar:
    st.markdown("## 🔍 Discovery Engine")
    st.markdown(
        "<p style='font-size:0.82rem;opacity:0.85;margin-top:-8px;'>"
        "Google Photos · Photo Retrieval Research</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    for key, label in NAV_ITEMS:
        is_active = st.session_state["page"] == key
        if st.button(
            label, key=f"nav_{key}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["page"] = key
            st.rerun()

    st.markdown("---")
    platforms = set(r.get("platform") for r in REVIEWS)
    st.markdown(f"**{len(REVIEWS)}** reviews collected")
    st.markdown(f"**{len(SURVEY)}** survey responses")
    st.markdown(f"**{len(platforms)}** platforms")

# ─────────────────────────────────────────────
# 8. PAGE: DASHBOARD
# ─────────────────────────────────────────────
def page_dashboard():
    st.markdown("# 📊 Dashboard")
    st.markdown("Overview of the Google Photos photo-retrieval research corpus.")

    # ── Metrics row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h2>{len(REVIEWS)}</h2><p>Public Reviews</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><h2>{len(SURVEY)}</h2><p>Survey Responses</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        platforms = set(r.get("platform") for r in REVIEWS)
        st.markdown(
            f'<div class="metric-card"><h2>{len(platforms)}</h2><p>Platforms</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        opp = compute_opportunity_areas()
        st.markdown(
            f'<div class="metric-card"><h2>{len(opp)}</h2><p>Opportunity Areas</p></div>',
            unsafe_allow_html=True,
        )

    # ── Reviews by Platform ──
    st.markdown('<div class="section-hdr">Reviews by Platform</div>', unsafe_allow_html=True)
    plat_counts = Counter(r.get("platform", "Unknown") for r in REVIEWS)
    for plat, cnt in plat_counts.most_common():
        pct = cnt / len(REVIEWS)
        st.markdown(f"**{plat}** — {cnt} reviews")
        st.progress(pct)

    # ── Affinity Mapping ──
    st.markdown('<div class="section-hdr">Affinity Mapping</div>', unsafe_allow_html=True)
    with st.expander("View Thematic Coding Methodology"):
        st.markdown("""
**How themes were identified:**

Each review was manually read and tagged with 1–3 themes from a taxonomy that emerged from the data itself. The taxonomy was developed through affinity mapping: grouping similar complaints, requests, and descriptions into clusters, then naming each cluster.

**Theme taxonomy (8 retrieval-focused themes):**

- **vague_memory_retrieval** — User remembers visual/emotional context but can't formulate the right search query
- **search_accuracy** — Search returns wrong, incomplete, or too many results
- **ai_search_regression** — Gemini/Ask Photos making search worse, slower, or less reliable
- **face_recognition** — Face grouping errors, misidentification, poor recognition edge cases
- **temporal_navigation** — Can't navigate by time, wrong timestamps, no date-range filtering
- **screenshot_document_search** — Can't find screenshots, documents, receipts, medical photos
- **location_search** — Location-based search gaps, missing GPS, no indoor location inference
- **organization_friction** — Albums, folders, archiving, tagging, manual effort

**Coding rules:** Each review gets 1–3 themes. Sentiment is tagged as positive/negative/mixed/neutral. User segment is inferred from context (e.g., mentions of children → "family organizer", mentions of camera gear → "photographer").
        """)

    # ── Opportunity Areas ──
    st.markdown(
        '<div class="section-hdr">Opportunity Areas Ranked by Impact</div>',
        unsafe_allow_html=True,
    )
    for area in compute_opportunity_areas():
        plat_str = ", ".join(f"{p}: {c}" for p, c in sorted(area["platforms"].items(), key=lambda x: -x[1]))
        st.markdown(f"""
        <div class="opp-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <h4>{area['label']}</h4>
                <span class="score">{area['impact']}/10</span>
            </div>
            <div style="color:#5F6368;font-size:0.85rem;margin-bottom:8px;">{area['description']}</div>
            <div style="color:#9AA0A6;font-size:0.75rem;">Evidence: {area['evidence_count']} reviews · {plat_str}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(area["impact"] / 10)


# ─────────────────────────────────────────────
# 9. PAGE: ASK INSIGHTS — PUBLIC REVIEWS
# ─────────────────────────────────────────────
def page_public_reviews():
    st.markdown("# 💬 Ask Insights from Public Reviews")
    st.markdown(
        "Ask any question about Google Photos retrieval problems. "
        "The engine searches the review corpus and synthesizes an AI-grounded answer."
    )

    # ── Suggested questions ──
    with st.expander("💡 Sample questions to try"):
        st.markdown("""
- What kinds of old photos do users struggle to retrieve?
- What information do people actually remember about a photo vs what they've forgotten?
- How do users formulate searches when their memory is incomplete?
- What are the biggest complaints about the Ask Photos / Gemini AI search?
- How does face recognition fail for family organizers?
- What types of 'functional' photos (documents, receipts, medical) are hardest to find?
- Where does location-based search break down?
- How do users feel about the trade-off between AI search and classic keyword search?
        """)

    query = st.text_input("🔎 Your question:", placeholder="e.g. What kinds of photos are hardest to find?")
    search_btn = st.button("Search & Synthesize", type="primary")

    if search_btn and query:
        results = review_engine.search(query, top_k=15)
        if not results:
            st.warning("No matching reviews found. Try different keywords.")
            return

        # ── AI Synthesis ──
        api_key = get_api_key()
        if api_key:
            evidence = "\n\n".join(
                f"[Review {r['id']} | {r['platform']} | {r['sentiment']}]: \"{r['text']}\""
                for r in results
            )
            user_msg = f"QUESTION: {query}\n\nREVIEWS:\n{evidence}"
            with st.spinner("Synthesizing AI answer..."):
                answer = call_groq(REVIEW_SYSTEM_PROMPT, user_msg, api_key)
            if answer.startswith("__ERROR__"):
                st.error(f"LLM error: {answer}")
            else:
                st.markdown(
                    f'<div class="ai-answer"><b>🤖 AI-Synthesized Answer</b><br><br>{answer}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Add your Groq API key in Streamlit Cloud Secrets (`GROQ_API_KEY`) to enable AI synthesis.")

        # ── Source Reviews ──
        st.markdown(
            f'<div class="section-hdr">Source Reviews ({len(results)} matches)</div>',
            unsafe_allow_html=True,
        )
        for r in results:
            render_review_card(r)

        # ── Theme distribution of results ──
        theme_dist = Counter()
        for r in results:
            for t in r.get("themes", []):
                theme_dist[t] += 1
        if theme_dist:
            st.markdown('<div class="section-hdr">Theme Distribution in Results</div>', unsafe_allow_html=True)
            for t, c in theme_dist.most_common():
                label = THEME_LABELS.get(t, t)
                st.markdown(f"**{label}** — {c} reviews")
                st.progress(c / len(results))


# ─────────────────────────────────────────────
# 10. PAGE: ASK INSIGHTS — USER SURVEY
# ─────────────────────────────────────────────
def page_user_survey():
    st.markdown("# 📋 Ask Insights from User Survey")

    if not SURVEY:
        st.info(
            "📂 **No survey data loaded yet.** Upload your survey data as `data/survey.json` "
            "and redeploy. The survey page will light up with search and AI synthesis over "
            "your primary research data."
        )
        st.markdown("---")
        st.markdown("### Expected format for `survey.json`")
        st.code("""{
  "id": 1,
  "demographics": { "gender": "Male", "age_group": "23-25", ... },
  "usage_behavior": { ... },
  "barriers_ratings": { "barrier_name": 3.0 },
  "open_ended_field": "User's verbatim response",
  "searchable_text": "All text fields concatenated for TF-IDF search"
}""", language="json")
        return

    # If survey data is loaded:
    st.markdown(f"**{len(SURVEY)}** survey responses loaded.")

    query = st.text_input("🔎 Your question:", placeholder="e.g. What barriers do users face when retrieving old photos?")
    search_btn = st.button("Search & Synthesize", type="primary")

    if search_btn and query and survey_engine:
        results = survey_engine.search(query, top_k=15)
        if not results:
            st.warning("No matching survey responses found.")
            return

        api_key = get_api_key()
        if api_key:
            evidence = "\n\n".join(
                f"[Respondent {r.get('id','')}]: {r.get('searchable_text','')[:500]}"
                for r in results
            )
            user_msg = f"QUESTION: {query}\n\nSURVEY DATA:\n{evidence}"
            with st.spinner("Synthesizing AI answer..."):
                answer = call_groq(SURVEY_SYSTEM_PROMPT, user_msg, api_key)
            if answer.startswith("__ERROR__"):
                st.error(f"LLM error: {answer}")
            else:
                st.markdown(
                    f'<div class="ai-answer"><b>🤖 AI-Synthesized Answer</b><br><br>{answer}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Add your Groq API key in Streamlit Cloud Secrets to enable AI synthesis.")

        st.markdown(f'<div class="section-hdr">Matching Responses ({len(results)})</div>', unsafe_allow_html=True)
        for r in results:
            st.markdown(
                f"""<div class="review-card">
                    <div style="color:#1A73E8;font-weight:600;font-size:0.85rem;">Respondent {r.get('id','')}</div>
                    <div style="color:#202124;font-size:0.9rem;margin-top:6px;">{r.get('searchable_text','')[:400]}...</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# 11. PAGE: REVIEW EXPLORER
# ─────────────────────────────────────────────
def page_explorer():
    st.markdown("# 🗂 Review Explorer")
    st.markdown("Filter and browse the full review corpus.")

    # ── Filters ──
    c1, c2, c3, c4 = st.columns(4)
    platforms = sorted(set(r.get("platform", "Unknown") for r in REVIEWS))
    all_themes = sorted(set(t for r in REVIEWS for t in r.get("themes", [])))
    sentiments = sorted(set(r.get("sentiment", "unknown") for r in REVIEWS))
    segments = sorted(set(r.get("user_segment", "unknown") for r in REVIEWS))

    with c1:
        sel_platforms = st.multiselect("Platform", platforms, default=platforms)
    with c2:
        sel_themes = st.multiselect("Theme", all_themes, default=all_themes)
    with c3:
        sel_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    with c4:
        sel_segments = st.multiselect("User Segment", segments, default=segments)

    filtered = [
        r for r in REVIEWS
        if r.get("platform") in sel_platforms
        and r.get("sentiment") in sel_sentiments
        and r.get("user_segment") in sel_segments
        and any(t in sel_themes for t in r.get("themes", []))
    ]

    st.markdown(f"**Showing {len(filtered)} of {len(REVIEWS)} reviews**")
    st.markdown("---")

    for r in filtered:
        render_review_card(r)


# ─────────────────────────────────────────────
# 12. PAGE: COMPARISON MATRIX
# ─────────────────────────────────────────────
def page_comparison():
    st.markdown("# 📈 Comparison Matrix")
    st.markdown("Interactive scatter plot: **Impact vs Evidence Volume** for each opportunity area.")

    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.error("Install altair and pandas: `pip install altair pandas`")
        return

    areas = compute_opportunity_areas()
    if not areas:
        st.warning("No opportunity areas computed.")
        return

    df = pd.DataFrame(areas)

    chart = (
        alt.Chart(df)
        .mark_circle(size=200, opacity=0.85)
        .encode(
            x=alt.X("evidence_count:Q", title="Evidence Volume (# reviews)", scale=alt.Scale(zero=True)),
            y=alt.Y("impact:Q", title="Impact Score (0–10)", scale=alt.Scale(domain=[0, 10.5])),
            color=alt.Color(
                "label:N", title="Opportunity Area",
                scale=alt.Scale(
                    range=["#1A73E8", "#EA4335", "#34A853", "#FBBC04", "#8E24AA", "#00ACC1", "#FF7043", "#78909C"]
                ),
            ),
            tooltip=["label:N", "impact:Q", "evidence_count:Q", "description:N"],
        )
        .properties(width=700, height=450)
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_legend(labelFontSize=11, titleFontSize=12)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # ── Summary table ──
    st.markdown('<div class="section-hdr">Summary Table</div>', unsafe_allow_html=True)
    summary_df = df[["label", "impact", "evidence_count"]].rename(
        columns={"label": "Opportunity Area", "impact": "Impact Score", "evidence_count": "Evidence Count"}
    )
    st.dataframe(summary_df.sort_values("Impact Score", ascending=False), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# 13. ROUTER
# ─────────────────────────────────────────────
if st.session_state["page"] == "dashboard":
    page_dashboard()
elif st.session_state["page"] == "public_reviews":
    page_public_reviews()
elif st.session_state["page"] == "user_survey":
    page_user_survey()
elif st.session_state["page"] == "explorer":
    page_explorer()
elif st.session_state["page"] == "comparison":
    page_comparison()
