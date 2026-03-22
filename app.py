"""
GenAI Report Agent — Data Reply Take-Home Challenge
Author: Dr. Yasser Mustafa
Built using PipeScraper (open-source) + LangChain + Groq + Streamlit
"""

import streamlit as st
import threading
import time
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.collector import collect_and_report
from src.chat import build_chat_chain
from src.report_store import load_latest_report, load_all_reports

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Report Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .report-card {
        background: #f8f9fa;
        border-left: 4px solid #0f3460;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .takeaway-item {
        background: #e8f4f8;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #17a2b8;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-running { background: #d4edda; color: #155724; }
    .status-idle    { background: #fff3cd; color: #856404; }
    .chat-msg-user      { background: #0f3460; color: white; padding: 0.8rem 1rem; border-radius: 12px 12px 2px 12px; margin: 0.4rem 0; }
    .chat-msg-assistant { background: #e9ecef; color: #333;  padding: 0.8rem 1rem; border-radius: 12px 12px 12px 2px; margin: 0.4rem 0; }
    .entity-tag { display: inline-block; background: #17a2b8; color: white; padding: 0.15rem 0.5rem; border-radius: 12px; margin: 0.15rem; font-size: 0.78rem; }
    .topic-tag  { display: inline-block; background: #6c757d; color: white; padding: 0.15rem 0.5rem; border-radius: 12px; margin: 0.15rem; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scheduler_running" not in st.session_state:
    st.session_state.scheduler_running = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None


def scheduler_loop(interval_seconds: int = 3600, topic: str = "AI Regulation", sources: list = None):
    """Run collect_and_report every `interval_seconds`."""
    if sources is None:
        sources = ["BBC Technology", "Google News"]
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    
    while st.session_state.get("scheduler_running", False):
        time.sleep(interval_seconds)
        if not st.session_state.get("scheduler_running", False):
            break
        try:
            collect_and_report(topic=topic, sources=sources)
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            st.session_state.chat_chain = None
        except Exception as e:
            print(f"[Scheduler Error] {e}")


# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🤖 GenAI Report Agent</h1>
    <p style="margin:0.5rem 0 0; opacity:0.85;">
        Autonomous news intelligence powered by <strong>PipeScraper</strong> · 
        <strong>LangChain</strong> · <strong>Groq (Llama-3)</strong>
    </p>
    <p style="margin:0.3rem 0 0; font-size:0.8rem; opacity:0.65;">
        Built by Dr. Yasser Mustafa
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Controls ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Agent Controls")

    topic = st.selectbox(
        "📰 Monitoring Topic",
        ["AI Regulation", "Technology Updates", "UK Economy", "Cybersecurity", "Climate & Energy"],
        index=0,
    )

    sources = st.multiselect(
        "🌐 News Sources",
        ["BBC Technology", "BBC Science", "Google News", "UK Gov"],
        default=["BBC Technology", "Google News"],
    )

    interval_label = st.selectbox(
        "⏱ Collection Interval",
        ["Every Hour (production)", "Every 5 mins (demo)","Every 1 min (demo)", "Manual only"],
        index=2,
    )
    interval_map = {
        "Every Hour (production)": 3600,
        "Every 5 mins (demo)": 300,
        "Every 1 min (demo)": 60,
        "Manual only": None,
    }
    interval = interval_map[interval_label]

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Agent", use_container_width=True, type="primary"):
            if not st.session_state.scheduler_running and interval:
                with st.spinner("Starting agent & collecting first report..."):
                    try:
                        collect_and_report(topic=topic, sources=sources)
                        st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                        st.session_state.chat_chain = None
                    except Exception as e:
                        st.error(f"Error: {e}")

                st.session_state.scheduler_running = True
                from streamlit.runtime.scriptrunner import add_script_run_ctx
                t = threading.Thread(
                    target=scheduler_loop, args=(interval, topic, sources), daemon=True
                )
                add_script_run_ctx(t)
                t.start()
                st.success("Agent started! It will now collect in the background.")
                st.rerun()

    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.scheduler_running = False
            st.info("Agent stopped.")

    if st.button("🔄 Collect Now", use_container_width=True):
        with st.spinner("Collecting & generating report..."):
            try:
                collect_and_report(topic=topic, sources=sources)
                st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
                st.session_state.chat_chain = None  # reset chain to pick up new report
                st.success("✅ Report generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    status_cls = "status-running" if st.session_state.scheduler_running else "status-idle"
    status_txt = "🟢 Running" if st.session_state.scheduler_running else "🟡 Idle"
    st.markdown(f'<span class="status-badge {status_cls}">{status_txt}</span>', unsafe_allow_html=True)

    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")

    st.divider()
    st.markdown("**🔗 Open Source**")
    st.markdown("[PipeScraper](https://github.com/Yasser03/pipescraper) — pipe-based scraping")
    st.markdown("[PipeFrame](https://github.com/Yasser03/pipeframe) — verb-based data ops")

# ─── Main Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Latest Report", "💬 Chat Interface", "📁 Report History"])

# ════════════════════════════════════════════════════════════
# TAB 1 — LATEST REPORT
# ════════════════════════════════════════════════════════════
def render_tab1():
    report = load_latest_report()

    if report is None:
        st.info("No reports yet. Click **🔄 Collect Now** in the sidebar to generate the first report.")
    else:
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("📰 Articles Collected", report.get("article_count", "—"))
        col_meta2.metric("🕐 Generated At", report.get("timestamp", "—")[-8:] if report.get("timestamp") else "—")
        col_meta3.metric("🏷 Topic", report.get("topic", "—"))

        st.subheader("📝 Summary")
        st.markdown(f'<div class="report-card">{report.get("summary", "")}</div>', unsafe_allow_html=True)

        st.subheader("🎯 Key Takeaways")
        for i, tk in enumerate(report.get("takeaways", []), 1):
            st.markdown(f'<div class="takeaway-item">✅ <strong>{i}.</strong> {tk}</div>', unsafe_allow_html=True)

        col_e, col_t = st.columns(2)
        with col_e:
            st.subheader("🏢 Organisations Mentioned")
            entities_html = " ".join(
                f'<span class="entity-tag">{e}</span>'
                for e in report.get("entities", [])
            )
            st.markdown(entities_html or "_None detected_", unsafe_allow_html=True)

        with col_t:
            st.subheader("🔖 Key Topics")
            topics_html = " ".join(
                f'<span class="topic-tag">{t}</span>'
                for t in report.get("key_topics", [])
            )
            st.markdown(topics_html or "_None detected_", unsafe_allow_html=True)

        st.divider()
        with st.expander("📰 Source Articles"):
            for art in report.get("articles", []):
                st.markdown(f"- [{art.get('title', art.get('url',''))}]({art.get('url','')})")

with tab1:
    if st.session_state.get("scheduler_running", False) and interval:
        st.fragment(run_every=interval)(render_tab1)()
    else:
        render_tab1()


# ════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ════════════════════════════════════════════════════════════
def render_tab2():
    st.subheader("💬 Chat with the Report Agent")
    st.caption("Ask anything about the latest collected news. The agent grounds its answers in the most recent report.")

    report_for_chat = load_latest_report()

    if report_for_chat is None:
        st.warning("Generate a report first before chatting.")
    else:
        # Lazy-build the chain once per session / after new report
        if st.session_state.chat_chain is None:
            with st.spinner("Initialising chat chain..."):
                st.session_state.chat_chain = build_chat_chain(report_for_chat)

        # Render history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Your question:", placeholder="What's happening in AI today?")
            submitted = st.form_submit_button("Send →")

        if submitted and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_chain.invoke({
                        "question": user_input,
                        "history": "".join([f"{msg['role']}: {msg['content']}\n" for msg in st.session_state.chat_history[:-1]]),
                    })
                    answer = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

with tab2:
    if st.session_state.get("scheduler_running", False) and interval:
        st.fragment(run_every=interval)(render_tab2)()
    else:
        render_tab2()


# ════════════════════════════════════════════════════════════
# TAB 3 — REPORT HISTORY
# ════════════════════════════════════════════════════════════
def render_tab3():
    st.subheader("📁 All Generated Reports")
    all_reports = load_all_reports()

    if not all_reports:
        st.info("No reports found yet.")
    else:
        for r in reversed(all_reports):
            with st.expander(f"📋 {r.get('timestamp', 'Unknown')} — {r.get('topic', '')} ({r.get('article_count', 0)} articles)"):
                st.markdown(f"**Summary:** {r.get('summary', '')}")
                st.markdown("**Takeaways:**")
                for tk in r.get("takeaways", []):
                    st.markdown(f"- {tk}")

with tab3:
    if st.session_state.get("scheduler_running", False) and interval:
        st.fragment(run_every=interval)(render_tab3)()
    else:
        render_tab3()
