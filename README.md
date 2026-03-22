# 🤖 GenAI Report Agent

> **Built by Dr. Yasser Mustafa**  

An autonomous GenAI system that collects news articles from the web, generates structured intelligence reports on a schedule, and lets users chat with the results through a Streamlit interface.

---

## 🏗 Architecture at a Glance

```text
Web Sources (BBC, Google News)
        │
        ▼
  [PipeScraper](https://github.com/Yasser03/pipescraper) (pipe-based scraping)
        │  FetchLinks >> ExtractArticles >> FilterArticles >> ToDataFrame
        ▼
  Report Generator (LangChain + Groq / Llama-3.3-70B)
        │  Summarisation · Takeaways · Entity Extraction · Topic Tagging
        ▼
  JSON Report Store  ←──────────────────────────────┐
        │                                            │
        ▼                                        APScheduler
  Streamlit Interface                           (hourly cron)
   ├── 📊 Latest Report tab
   ├── 💬 Chat Interface tab   ← grounded on latest report
   └── 📁 Report History tab
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Yasser03/<repo>
cd genai_report_agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file in your root directory:
```env
GROQ_API_KEY="your_groq_api_key_here"
```

> Get a free Groq API key at https://console.groq.com — no credit card required.

### 3. Run Headless (CLI / Production)

If you don't want the UI, you can run the agent headlessly:
```bash
# Single run
python run_agent.py --once --topic "AI Regulation"

# Hourly scheduling (production)
python run_agent.py --topic "AI Regulation" --interval 60

# Every 5 minutes (demo)
python run_agent.py --topic "Technology Updates" --interval 5
```

---

## 🖥️ Streamlit UI Guide (Step-by-Step)

If you want the full visual experience with automated visual refreshing, use the Streamlit Dashboard:

```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### Step 1: Configure Your Agent
In the **⚙️ Agent Controls** sidebar, adjust your settings:
- **Monitoring Topic:** The subject to focus the news collection on (e.g., AI Regulation, Cybersecurity).
- **News Sources:** Select which online sources to dynamically scrape (e.g., BBC Technology, Google News).
- **Collection Interval:** Choose the scheduler timing step (e.g., Every 1 min, 5 mins, or Hourly).

### Step 2: Use the Sidebar Buttons Explained
- **`▶ Start Agent`**: **Starts the autonomous background scheduler.** When clicked, the app runs an immediate synchronous scrape (updating your screen instantly) and then forks a background thread. From then on, it will fetch articles every `interval` without you doing anything. 
- **`⏹ Stop`**: **Gracefully stops the scheduler.** The background thread will exit after its current sleep cycle finishes, halting further automation.
- **`🔄 Collect Now`**: **Manual Override.** Bypasses the scheduler entirely to perform an immediate, one-off fetch-generation-update cycle.

### Step 3: Explore the Dashboard Tabs
Thanks to native Streamlit fragments, all these tabs **automatically refresh** as long as the background agent is running!

- **`📊 Latest Report`**: View the newest generated intelligence. You'll see articles collected, timestamp, generation topic, an AI-curated summary, distinct takeaways, and any extracted organisations/topics.
- **`💬 Chat Interface`**: Safe, grounded chat interaction. The LangChain logic is injected with *only* the contents of the latest report. If the agent runs in the background and collects a *new* report, the chat cycle is safely reset and re-hooks to the newly generated data instantly!
- **`📁 Report History`**: Browse a persistent history log of all past reports generated across all your sessions.

---

## 📦 Open-Source Dependencies

This exact technical assignment was accelerated by my custom open-source libraries:
- **[PipeScraper](https://github.com/Yasser03/pipescraper)** — Open-source pipe-based web scraping built on [Trafilatura](https://github.com/adbar/trafilatura) and [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). The agent uses `FetchGoogleNews` and `ExtractArticles` in pipelines.
- **[PipeFrame](https://github.com/Yasser03/pipeframe)** — Verb-based Pandas operations that handle declarative logic steps.

Other integrations include:
- **LangChain** — orchestration and prompt chaining
- **langchain-groq** — Groq inference backend (Llama-3.3-70B)
- **Streamlit** — native UI elements and fragments architecture
- **APScheduler** — production scheduling

---

## 📖 Extended Documentation

This repository contains two detailed documents explaining the engineering decisions, architecture, and design principles behind the GenAI Report Agent:

1. **[Technical Write-Up](Technical_Writeup_YasserMustafa.md)**  
   A comprehensive technical breakdown of the system architecture, structured LLM prompting strategies, grounded conversational chat setup, and the specific alternatives evaluated during development.

2. **[GenAI Report Agent Article](genai_report_agent_article.md)**  
   A professionally formatted, creative Medium-style article walking through the end-to-end integration of pipe-based scraping, LLM synthesisation, and RAG-lite retrieval from first principles.

---

## 🗂 Project Structure

```text
genai_report_agent/
├── app.py                  # Streamlit multi-tab UI
├── run_agent.py            # Headless CLI runner with APScheduler
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── collector.py        # [PipeScraper](https://github.com/Yasser03/pipescraper) data collection + Groq report generation
│   ├── chat.py             # LangChain conversational chain (grounded on report)
│   └── report_store.py     # JSON persistence layer
└── reports/
```

---

## 📄 License

MIT — See LICENSE.

---
*Built with ❤️ by Dr. Yasser Mustafa — AI & Data Science Specialist*
