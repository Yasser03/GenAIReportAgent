# GenAI Report Agent - Technical Write-Up
  
**Dr. Yasser Mustafa | March 2026**

## 1. Problem Understanding
The challenge requires building an autonomous GenAI system capable of three core functions:
- Collect and clean text content from trusted web sources (BBC News, UK Government) on a recurring schedule.
- Generate a structured intelligence report — including a 100-150 word summary, key takeaways, and extracted entities — every hour without human intervention.
- Provide a conversational interface through which users can query the collected information in natural language, including handling vague, open-ended questions.

The fundamental design challenge is ensuring the system is both autonomous (runs on a schedule without user prompting) and grounded (the chatbot must respond based on what was actually collected, not hallucinated content). This requires careful integration of scraping, LLM orchestration, and state management.

## 2. Solution Overview
The system is implemented as a modular Python application with four layers:

### 2.1 Data Collection — [PipeScraper](https://github.com/Yasser03/pipescraper)
Article collection is handled by **[PipeScraper](https://github.com/Yasser03/pipescraper)**, an open-source pipe-based news scraping library built on [Trafilatura](https://github.com/adbar/trafilatura) and [newspaper4k](https://github.com/AndyTheFactory/newspaper4k). The library exposes a composable `>>` operator API:
`FetchGoogleNews(search=["AI regulation"], period="1d") >> ExtractArticles(workers=4) >> ToDataFrame()`

[PipeScraper](https://github.com/Yasser03/pipescraper) provides: parallel article extraction, Google News search with automatic URL decoding (bypassing the consent wall), robots.txt compliance, built-in deduplication, and native DataFrame export. The system falls back to BBC RSS feeds if the primary pipeline is unavailable.

### 2.2 Report Generation — LangChain + Groq
Collected article text is passed to a structured LangChain prompt chain backed by Groq's Llama-3.3-70B inference API. The model is instructed to return a strict JSON object containing: a 100-150 word summary, 3-5 key takeaways, mentioned organisations/entities, and key topic tags. JSON parsing is hardened against markdown fence contamination.

### 2.3 Scheduling — APScheduler
In CLI/production mode, an APScheduler `BlockingScheduler` fires the collection-report cycle at the configured interval (default: 60 minutes). In Streamlit mode, a daemon thread handles background scheduling, with a manual trigger always available in the sidebar.

### 2.4 Conversational Interface — Streamlit + LangChain
The Streamlit app provides a three-tab interface: Latest Report (structured display), Chat Interface, and Report History. The chat chain is constructed by injecting the full serialised report as a system-level grounding context, ensuring the LLM grounds its answers in collected data. Vague queries such as "What's happening?" or "Any news?" are handled via a prompt instruction that redirects them to report summarisation.

## 3. Key GenAI Concepts Used

| Concept                            | Where Applied                                                 | Why                                                                              |
| ---------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Structured Prompting**           | Report generation prompt forces strict JSON schema output     | Eliminates post-hoc parsing ambiguity; ensures reproducible structured outputs   |
| **Grounded Generation (RAG-lite)** | Full report injected as system context for the chat chain     | Prevents hallucination; ensures chat responses are evidence-based                |
| **Instruction Hierarchy**          | System prompt sets constraints; user turn carries query       | Separates behavioural rules from user intent — standard LLM safety practice      |
| **Temperature Control**            | temp=0.3 for reports; temp=0.5 for chat                       | Lower temp = more factual reports; slightly higher = natural conversational tone |
| **Entity Extraction**              | LLM identifies organisations and key actors from article text | Avoids brittle NER pipelines; LLM context awareness yields richer extraction     |

## 4. Alternatives Considered

### LLM Provider
OpenAI GPT-4o was the obvious choice given its capability ceiling, but Groq's Llama-3.3-70B was selected for three reasons: it is free at reasonable throughput, latency is significantly lower (Groq's custom inference hardware), and it avoids vendor lock-in. For a production deployment, the chain is LLM-agnostic via LangChain's abstraction layer and could be swapped to GPT-4o, Claude 3.5, or a local Ollama instance with a one-line change.

### Scraping Layer
BeautifulSoup + Requests was the baseline option specified in the brief. [PipeScraper](https://github.com/Yasser03/pipescraper) was chosen instead because: (1) it wraps [Trafilatura](https://github.com/adbar/trafilatura), which is the state-of-the-art in boilerplate-free article extraction, (2) it provides a clean declarative API that makes the pipeline logic self-documenting, and (3) it natively integrates Google News search — enabling keyword-driven collection rather than fixed-URL scraping.

### Chat Framework
LlamaIndex was considered for its native RAG capabilities. LangChain was chosen because the grounding requirement is lightweight (one report, not a corpus) and LangChain's prompt/chain abstraction is more ergonomic for this pattern. Full vector RAG (ChromaDB + embeddings) would be the natural next step if the report history grew large.

### Scheduling
A simple schedule library loop was the minimal option. APScheduler was chosen for production readiness: it supports cron-style scheduling, misfire handling, and job persistence — important when the agent is expected to run continuously. Airflow or Prefect would be appropriate for a distributed production system.

## 5. Open-Source Foundation
**[PipeScraper](https://github.com/Yasser03/pipescraper)** is an open-source Python library developed by me, Dr. Yasser Mustafa, and published on GitHub and PyPI. I have also created a broader ecosystem of open-source packages alongside it, including **[PipeFrame](https://github.com/Yasser03/pipeframe)** (verb-based Pandas operations), **[PipePlotly](https://github.com/Yasser03/pipeplotly)** (Grammar of Graphics visualisation), and **openclawpy** (OpenClaw Gateway SDK). The IIC (Incident Intelligence Collector) platform — a production Streamlit news intelligence pipeline using Groq and LangChain — provided direct architectural precedent for this submission. This challenge solution demonstrates how open-source tooling I developed independently can be applied to real production problems.

---
**Dr. Yasser Mustafa**  
*Lead Data Scientist — PhD Theoretical Physics — 10+ years production AI*  
[GitHub](https://github.com/Yasser03) | [Medium](https://medium.com/@yasser.mustafa) | yasser.mustafan@gmail.com
