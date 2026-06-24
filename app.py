import os
import json
import base64
import subprocess
import sys
from pathlib import Path
import numpy as np
import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = "vector_store"
EMBED_MODEL      = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
TOP_K            = 5
MAX_HISTORY_PAIRS = 4

# ── Lucide-style SVG icon library ──────────────────────────────────────────────
# Stroke-based, 18 × 18, rounded caps — same design language as IBM Carbon / Apple SF Symbols
def icon(path_d, extra="", size=18):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" {extra}>{path_d}</svg>'
    )

ICONS = {
    "home":       icon('<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>'),
    "chart":      icon('<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'),
    "cpu":        icon('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
    "heart":      icon('<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>'),
    "bus":        icon('<rect x="1" y="3" width="15" height="13" rx="2"/><path d="M16 8h4l3 3v5h-7V3"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>'),
    "award":      icon('<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11"/>'),
    "file":       icon('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/>'),
    "shield":     icon('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
    "book":       icon('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'),
    "palette":    icon('<circle cx="13.5" cy="6.5" r=".5"/><circle cx="17.5" cy="10.5" r=".5"/><circle cx="8.5" cy="7.5" r=".5"/><circle cx="6.5" cy="12.5" r=".5"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>'),
    "send":       icon('<line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>'),
    "trash":      icon('<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>'),
    "bot":        icon('<rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/>'),
    "sparkle":    icon('<path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/><path d="M5 3l.75 2.25L8 6l-2.25.75L5 9l-.75-2.25L2 6l2.25-.75z"/><path d="M19 15l.75 2.25L22 18l-2.25.75L19 21l-.75-2.25L16 18l2.25-.75z"/>'),
    "phone":      icon('<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.84 12 19.79 19.79 0 0 1 1.77 3.4 2 2 0 0 1 3.74 1h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.91 8.96a16 16 0 0 0 6.29 6.29l1.06-1.06a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>'),
    "chevron":    icon('<polyline points="9 18 15 12 9 6"/>'),
    "dot_circle": icon('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3" fill="currentColor"/>'),
}

# ── Topic definitions ──────────────────────────────────────────────────────────
TOPICS = [
    ("home",    "Hostel & Accommodation",     "What are the hostel rules and facilities at Avantika University?"),
    ("chart",   "Grades & CGPA",              "Explain the grading system and how CGPA is calculated."),
    ("cpu",     "B.Tech Program",             "What are the rules and structure for the B.Tech program?"),
    ("heart",   "Health Centre",              "What are the health centre timings and services?"),
    ("bus",     "Transport & Bus Service",    "How do I register for the university bus service?"),
    ("award",   "Scholarships",               "How do I apply for a scholarship at Avantika University?"),
    ("file",    "Transcripts & Certificates", "How do I get my academic transcript or certificate?"),
    ("shield",  "Anti-Ragging",               "What is the anti-ragging policy and how to report ragging?"),
    ("book",    "Supplementary Exams",        "How do supplementary or backlog exams work?"),
    ("palette", "B.Des Program",              "Tell me about the Bachelor of Design (B.Des) program."),
]

SUGGESTED_Q = [
    "What is the CGPA cut-off for First Class with Distinction?",
    "What items are prohibited in the hostel?",
    "What are the ATKT rules for B.Tech students?",
    "How do I report ragging? What is the helpline number?",
    "What B.Tech specializations are offered?",
    "Who is the Vice Chancellor of Avantika University?",
]

SYSTEM_PROMPT = """You are AvantikaBot, the official AI student assistant for Avantika University, Ujjain, Madhya Pradesh (an MIT Group Institution).

Answer student queries based ONLY on the context from official university documents provided. Guidelines:
- Be friendly, accurate, and concise. Use bullet points for lists.
- If the context doesn't contain enough information, say so and direct to the relevant office.
- Mention contact numbers or offices when relevant.
- Anti-ragging helpline: 1800-180-5522 (24×7, free).
- Do not invent policies, numbers, or facts.

Respond in English unless the student writes in Hindi."""

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AvantikaBot — Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_logo_b64():
    p = Path("assets/logo.svg")
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None

LOGO_B64 = load_logo_b64()
LOGO_SRC  = f"data:image/svg+xml;base64,{LOGO_B64}" if LOGO_B64 else None

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, *::before, *::after {{ box-sizing: border-box; font-family: 'Inter', sans-serif; }}

/* ── Globals ── */
.stApp {{ background: #EEF3FF; }}
#MainMenu, footer, header, [data-testid="stToolbar"] {{ display: none !important; visibility: hidden !important; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(170deg, #002A75 0%, #001040 100%) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.18);
}}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {{ color: #A8C4E8 !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: #fff !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12) !important; }}

/* sidebar topic buttons */
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,.07) !important;
    color: #C8DCF4 !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    border-radius: 8px !important;
    width: 100% !important;
    text-align: left !important;
    padding: .42rem .85rem !important;
    font-size: .81rem !important;
    font-weight: 500 !important;
    margin-bottom: 2px !important;
    letter-spacing: .01em !important;
    transition: background .15s, border-color .15s, color .15s !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,102,0,.22) !important;
    border-color: rgba(255,102,0,.55) !important;
    color: #fff !important;
}}
[data-testid="stSidebar"] .stButton > button:active {{
    background: rgba(255,102,0,.35) !important;
}}

/* ── Header ── */
.au-header {{
    background: linear-gradient(135deg, #002A75 0%, #001040 100%);
    padding: 1rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,40,120,.28);
}}
.au-header img {{ height: 38px; flex-shrink: 0; }}
.au-header-divider {{
    width: 1px; height: 36px;
    background: rgba(255,255,255,.2);
    flex-shrink: 0;
}}
.au-header-text h1 {{
    margin: 0; font-size: 1.05rem; font-weight: 700;
    color: #fff; line-height: 1.2;
}}
.au-header-text p {{
    margin: 2px 0 0; font-size: .75rem; color: #7AAAD8;
}}
.au-badge {{
    margin-left: auto;
    background: rgba(255,102,0,.18);
    border: 1px solid rgba(255,102,0,.4);
    color: #FFAA70;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: .71rem;
    font-weight: 600;
    display: flex; align-items: center; gap: 5px;
    white-space: nowrap;
}}
.status-dot {{
    width: 7px; height: 7px;
    background: #22C55E;
    border-radius: 50%;
    display: inline-block;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{ border: none !important; background: transparent !important; padding: 0 !important; }}
[data-testid="stChatMessage"] .stChatMessageContent {{
    border-radius: 14px !important;
    padding: .85rem 1.1rem !important;
    font-size: .88rem !important;
    line-height: 1.6 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.08) !important;
    border: none !important;
}}
/* user bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stChatMessageContent {{
    background: linear-gradient(135deg, #002A75, #004ABB) !important;
    color: #fff !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 2px 12px rgba(0,42,117,.25) !important;
}}
/* assistant bubble */
[data-testid="stChatMessage"][data-testid*="assistant"] .stChatMessageContent {{
    background: #fff !important;
    color: #1A202C !important;
    border-radius: 4px 18px 18px 18px !important;
    border-left: 3px solid #E64400 !important;
}}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {{
    border-radius: 24px !important;
    border: 2px solid #002A75 !important;
    padding: .65rem 1.2rem !important;
    font-size: .88rem !important;
    background: #fff !important;
    transition: border-color .15s, box-shadow .15s !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: #E64400 !important;
    box-shadow: 0 0 0 3px rgba(230,68,0,.12) !important;
}}
[data-testid="stChatInput"] button {{
    background: #002A75 !important;
    border-radius: 50% !important;
    border: none !important;
}}
[data-testid="stChatInput"] button:hover {{
    background: #E64400 !important;
}}
[data-testid="stChatInput"] button svg {{ color: #fff !important; }}

/* ── Welcome card ── */
.welcome-card {{
    background: #fff;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
    border-top: 3px solid #E64400;
    display: flex; align-items: flex-start; gap: 1rem;
}}
.welcome-icon {{
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #002A75, #004ABB);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; color: #fff;
}}
.welcome-card h4 {{ color: #002A75; margin: 0 0 .3rem; font-size: .95rem; font-weight: 600; }}
.welcome-card p  {{ color: #64748B; margin: 0; font-size: .82rem; line-height: 1.55; }}

/* ── Suggested question chips ── */
.sq-chip button {{
    background: #fff !important;
    color: #002A75 !important;
    border: 1.5px solid #002A75 !important;
    border-radius: 20px !important;
    font-size: .79rem !important;
    padding: .32rem .9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all .15s !important;
}}
.sq-chip button:hover {{
    background: #002A75 !important;
    color: #fff !important;
    border-color: #002A75 !important;
}}

/* ── Source badge ── */
.src-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.src-badge {{
    display: inline-flex; align-items: center; gap: 4px;
    background: #FFF3E0; color: #B84400;
    border: 1px solid #FFB74D;
    padding: 3px 10px; border-radius: 10px;
    font-size: .71rem; font-weight: 600;
}}
.src-badge svg {{ flex-shrink: 0; }}

/* ── Sidebar structural ── */
.sb-logo-wrap {{
    text-align: center;
    padding: .5rem 1rem 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,.1);
    margin-bottom: .75rem;
}}
.sb-logo-wrap img {{ width: 75%; max-width: 160px; }}
.sb-section {{
    font-size: .65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .09em;
    color: rgba(255,255,255,.35) !important;
    padding: .6rem 0 .25rem;
}}
.sb-icon-row {{
    display: flex; align-items: center; gap: .55rem;
    padding: .55rem .75rem;
    border-radius: 8px;
    color: rgba(255,255,255,.6);
    font-size: .78rem; font-weight: 500;
}}
.sb-icon-row svg {{ flex-shrink: 0; opacity: .7; }}
.helpline-card {{
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 10px;
    padding: .75rem 1rem;
    margin-top: .75rem;
    font-size: .72rem;
    color: #8AAEDE;
    text-align: center; line-height: 1.65;
}}
.helpline-card strong {{ color: #fff; }}
.helpline-num {{ color: #FF9B6A; font-weight: 700; font-size: .85rem; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_resources():
    model    = SentenceTransformer(EMBED_MODEL)
    index    = faiss.read_index(os.path.join(VECTOR_STORE_DIR, "index.faiss"))
    metadata = json.loads(Path(os.path.join(VECTOR_STORE_DIR, "metadata.json")).read_text(encoding="utf-8"))
    return model, index, metadata


def retrieve(query, model, index, meta, k=TOP_K):
    emb = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)
    scores, ids = index.search(emb, k)
    return [
        {"text": meta[i]["text"], "source": meta[i]["source"], "score": float(s)}
        for s, i in zip(scores[0], ids[0])
        if i >= 0 and s > 0.20
    ]


def build_messages(query, chunks, history):
    ctx = "\n".join(f"[{n+1}] ({c['source']})\n{c['text']}" for n, c in enumerate(chunks))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history[-(MAX_HISTORY_PAIRS * 2):]
    msgs.append({"role": "user", "content": f"Context:\n{ctx}\n\nStudent question: {query}"})
    return msgs


def unique_sources(chunks):
    seen, out = set(), []
    for c in chunks:
        if c["source"] not in seen:
            seen.add(c["source"]); out.append(c["source"])
    return out


def ensure_kb():
    if not (Path(VECTOR_STORE_DIR,"index.faiss").exists() and
            Path(VECTOR_STORE_DIR,"metadata.json").exists()):
        with st.spinner("Building knowledge base for the first time (~1 min)…"):
            r = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
        if r.returncode != 0:
            st.error(f"Failed to build knowledge base:\n{r.stderr}"); st.stop()
        st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        if LOGO_SRC:
            st.markdown(
                f'<div class="sb-logo-wrap">'
                f'<img src="{LOGO_SRC}" style="filter:brightness(0) invert(1);" />'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="sb-logo-wrap"><span style="color:#fff;font-weight:700;font-size:1.3rem;">AU</span></div>', unsafe_allow_html=True)

        # About text
        st.markdown('<div class="sb-section">About</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#8AAEDE;font-size:.78rem;line-height:1.55;padding:0 .25rem;">'
            'Ask anything about academics, hostel, health, transport, scholarships, '
            'programs, and more.</p>',
            unsafe_allow_html=True,
        )

        # Quick topics
        st.markdown('<div class="sb-section">Quick Topics</div>', unsafe_allow_html=True)
        for icon_key, label, question in TOPICS:
            svg = ICONS.get(icon_key, "")
            # Show icon label above the button for visual reference
            st.markdown(
                f'<div class="sb-icon-row">{svg}'
                f'<span style="color:rgba(255,255,255,.5);font-size:.72rem;">{label}</span></div>',
                unsafe_allow_html=True,
            )
            if st.button(label, key=f"topic_{icon_key}"):
                st.session_state.pending = question
                st.rerun()

        st.markdown("---")

        # Clear button with icon
        st.markdown(
            f'<div class="sb-icon-row" style="color:rgba(255,255,255,.4);margin-bottom:4px;">'
            f'{ICONS["trash"]}<span style="font-size:.72rem;">Clear conversation</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Conversation", key="clear"):
            st.session_state.messages = []
            st.session_state.pop("pending", None)
            st.rerun()

        # Helpline card
        st.markdown(
            f'<div class="helpline-card">'
            f'{ICONS["phone"]}<br>'
            f'<strong>Anti-Ragging Helpline</strong><br>'
            f'<span class="helpline-num">1800-180-5522</span><br>'
            f'<span>24×7 &nbsp;|&nbsp; Free &nbsp;|&nbsp; Confidential</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ensure_kb()

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        st.error("**GROQ_API_KEY not set.** Add it to `.env`:\n```\nGROQ_API_KEY=gsk_...\n```")
        st.stop()

    render_sidebar()

    try:
        embed_model, faiss_index, metadata = load_resources()
    except Exception as e:
        st.error(f"Could not load knowledge base: {e}\n\nRun `python ingest.py` first."); st.stop()

    groq_client = Groq(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Header ──
    logo_html = (
        f'<img src="{LOGO_SRC}" alt="Avantika University" />'
        if LOGO_SRC else
        '<span style="color:#fff;font-size:1.1rem;font-weight:800;letter-spacing:-.5px;">avantika<br><small style="font-weight:400;font-size:.65rem;letter-spacing:.15em;">UNIVERSITY</small></span>'
    )
    st.markdown(f"""
    <div class="au-header">
        {logo_html}
        <div class="au-header-divider"></div>
        <div class="au-header-text">
            <h1>Student Knowledge Assistant</h1>
            <p>Powered by official university documents &nbsp;·&nbsp; MIT Group Institution</p>
        </div>
        <div class="au-badge">
            <span class="status-dot"></span>
            AI Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome (only when no messages) ──
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="welcome-card">
            <div class="welcome-icon">{ICONS["sparkle"]}</div>
            <div>
                <h4>Hello! I'm AvantikaBot</h4>
                <p>Ask me about exams, hostel, health, transport, scholarships, program rules,
                   faculty, or anything else about Avantika University.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested question chips
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_Q):
            with cols[i % 2]:
                with st.container():
                    st.markdown('<div class="sq-chip">', unsafe_allow_html=True)
                    if st.button(q, key=f"sq_{i}"):
                        st.session_state.pending = q
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat history ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                badges = "".join(
                    f'<span class="src-badge">'
                    f'{ICONS["file"]}&nbsp;{s}'
                    f'</span>'
                    for s in msg["sources"]
                )
                st.markdown(f'<div class="src-row">{badges}</div>', unsafe_allow_html=True)

    # ── Input ──
    user_input = st.session_state.pop("pending", None) or st.chat_input("Ask anything about Avantika University…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        chunks  = retrieve(user_input, embed_model, faiss_index, metadata)

        with st.chat_message("assistant"):
            messages = build_messages(user_input, chunks, history)
            response = ""
            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL, messages=messages,
                    max_tokens=900, temperature=0.3, stream=True,
                )
                placeholder = st.empty()
                for part in stream:
                    delta = part.choices[0].delta.content or ""
                    response += delta
                    placeholder.markdown(response + "▌")
                placeholder.markdown(response)
            except Exception as e:
                response = f"Connection error. Please try again.\n\n`{e}`"
                st.error(response)

            sources = unique_sources(chunks)
            if sources:
                badges = "".join(
                    f'<span class="src-badge">{ICONS["file"]}&nbsp;{s}</span>'
                    for s in sources
                )
                st.markdown(f'<div class="src-row">{badges}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", "content": response, "sources": sources,
        })


if __name__ == "__main__":
    main()
