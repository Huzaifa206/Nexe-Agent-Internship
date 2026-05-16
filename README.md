# 🤖 Nexe-Agent Internship — Agentic AI Developer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6C47FF?style=for-the-badge&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**6 production-grade Agentic AI applications built during the Nexe-Agent internship.**  
Each task is a fully self-contained Streamlit app showcasing a different pillar of agentic AI development.

[🌐 Portfolio](https://nexe-agent-internship-portfolio.vercel.app/) · [📧 Contact](mailto:huzaifaasiddiqui@gmail.com) · [💼 LinkedIn](https://www.linkedin.com/in/huzaifaahmedsiddiqui48/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Tasks](#-tasks)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Deployment](#-deployment)
- [Author](#-author)

---

## 🧠 Overview

This repository contains all **6 internship tasks** for the **Agentic AI Developer** role at [Nexe-Agent](https://www.linkedin.com/company/nexe-agent). Each task is built as an independent Streamlit application and demonstrates a core concept in agentic AI — from basic tool calling to a full multi-agent system.

All apps use **OpenRouter** as the LLM gateway with **Nvidia Nemotron-3-super-120b-a12b** (free tier) as the model, meaning the entire project runs at zero cost.

---

## 🗂 Tasks

### 🟢 Beginner

#### Task 1 — Tool-Calling AI Agent
> `task1.py`

An AI agent that dynamically calls Python functions based on user intent, returns structured JSON, and handles errors gracefully.

**Covers:** Function calling · JSON responses · Error handling

**Tools available to agent:**
- `get_weather(city)` — returns weather for Karachi, Lahore, Islamabad, Dubai, London
- `calculate(expression)` — evaluates any math expression including trig, log, sqrt
- `get_current_time()` — returns current UTC date and time

**Key concept:** The agentic loop — the model runs in a `while True` loop, calling tools until it reaches a final answer (`end_turn`).

---

#### Task 2 — AI Calculator Agent
> `task2.py`

A math-focused agent with persistent session memory, structured step-by-step output, and support for complex multi-step word problems.

**Covers:** Math operations · Memory · Structured output

**Tools available to agent:**
- `calculate(expression)` — full Python math library support
- `remember(label, value)` — saves a value to session memory
- `recall(label)` — retrieves a saved value by name
- `recall_all()` — returns all saved values
- `clear_memory()` — wipes memory store
- `solve_steps(problem)` — instructs agent to solve step by step

**Key concept:** `st.session_state` as a memory store that persists across turns in a conversation.

---

### 🟡 Intermediate

#### Task 3 — Multi-Tool Agent
> `task3.py`

An agent that combines live web search, local database persistence, and email dispatch in a single conversational interface.

**Covers:** Web search · Save to DB · Send email

**Tools available to agent:**
- `web_search(query)` — DuckDuckGo search, no API key needed
- `save_to_db(title, content, source)` — persists to SQLite
- `get_all_notes()` — retrieves all saved notes
- `delete_note(id)` — removes a note by ID
- `send_email(to, subject, body)` — Gmail SMTP

**Key concept:** Chaining tools — agent searches, then saves results, then optionally emails them, all in one request.

---

#### Task 4 — RAG Assistant
> `task4.py`

A Retrieval Augmented Generation system that lets users upload PDF or TXT documents, indexes them into a local vector store, and answers questions using only the document content.

**Covers:** Document upload · Vector store · Contextual answers

**Pipeline:**
```
Upload PDF/TXT → Parse → Chunk (500 words, 100 overlap) → Embed → ChromaDB
User Question  → Embed → Cosine Similarity Search → Top-K Chunks → LLM Answer
```

**Tools available to agent:**
- `retrieve_context(query, n_results)` — semantic search over ChromaDB
- `list_documents()` — shows indexed documents and chunk counts

**Key concept:** The difference between parametric knowledge (what the model knows) and retrieved knowledge (what's in your documents).

---

### 🔴 Advanced

#### Task 5 — Autonomous Business Agent
> `task5.py`

A fully autonomous agent that accepts a business goal, generates a multi-step execution plan, executes each step using appropriate tools, logs every action to SQLite, and produces a downloadable report.

**Covers:** Multi-step reasoning · Task planning · Execution logs

**Tools available to agent:**
- `create_plan(goal)` — LLM-generated step-by-step plan
- `web_search(query)` — market and competitor research
- `analyze_data(data, type)` — summary / SWOT / trends / recommendations
- `write_report(title, sections)` — structured markdown report
- `calculate_metric(formula, values)` — business KPI calculations
- `mark_step_done(step, name, summary)` — progress tracking

**Key concept:** The agent doesn't ask for permission between steps — it plans and executes autonomously until the goal is complete.

---

#### Task 6 — Multi-Agent System
> `task6.py`

A team of 4 specialized AI agents that collaborate to complete complex tasks. An Orchestrator agent delegates work to specialist agents, each with their own system prompt, tools, and area of expertise.

**Covers:** Separate agents · Communication layer · Task delegation

**Agent team:**

| Agent | Role | Tools |
|---|---|---|
| 🎯 Orchestrator | Receives request, plans, delegates | `delegate_to_*` tools |
| 🔍 Research Agent | Web search & information gathering | `research_web_search` |
| 📊 Analyst Agent | Data analysis & insight extraction | `calculate_metric` |
| ✍️ Writer Agent | Report writing & content creation | none (pure LLM) |

**Communication layer:** Every message between agents is posted to a shared communication bus (`st.session_state.comm_bus`) and displayed live in the sidebar.

**Key concept:** Separation of concerns — each agent is an independent LLM call with its own context, preventing the model from mixing roles.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM Gateway** | OpenRouter |
| **Model** | Nvidia Nemotron-3-super-120b-a12b (free) |
| **LLM SDK** | OpenAI Python SDK (OpenRouter-compatible) |
| **Web Search** | DDGS (DuckDuckGo Search) |
| **Vector Store** | ChromaDB |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Database** | SQLite (built-in Python) |
| **PDF Parsing** | pypdf |
| **Package Manager** | uv |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

```
nexe-agent-internship/
│
├── task1.py                  # Tool-Calling AI Agent
├── task2.py                  # AI Calculator Agent
├── task3.py                  # Multi-Tool Agent
├── task4.py                  # RAG Assistant
├── task5.py                  # Autonomous Business Agent
├── task6.py                  # Multi-Agent System
│
├── requirements.txt          # For Streamlit Cloud deployment
├── pyproject.toml            # uv project config
├── uv.lock                   # uv lockfile
├── .python-version           # Python version pin
├── .gitignore                # Excludes .env and .venv
└── README.md
```

Each task file is **fully self-contained** — agent logic and Streamlit UI live together in one file for simplicity.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [OpenRouter](https://openrouter.ai) API key (free)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Huzaifa206/nexe-agent-internship.git
cd nexe-agent-internship

# 2. Install uv (if not already installed)
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Mac/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv venv
uv add streamlit openai python-dotenv ddgs chromadb sentence-transformers pypdf

# 4. Add your API key
echo OPENROUTER_API_KEY=sk-or-your-key-here > .env
```

### Running any task

```bash
uv run streamlit run task1.py   # Tool-Calling Agent
uv run streamlit run task2.py   # Calculator Agent
uv run streamlit run task3.py   # Multi-Tool Agent
uv run streamlit run task4.py   # RAG Assistant
uv run streamlit run task5.py   # Business Agent
uv run streamlit run task6.py   # Multi-Agent System
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for all tasks
OPENROUTER_API_KEY=sk-or-your-key-here

# Optional — only for Task 3 email feature
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## ☁️ Deployment

Each task is deployed independently on **Streamlit Cloud** (free).

### Steps
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → set **Main file path** to e.g. `task1.py`
4. Go to **Advanced settings → Secrets** and add:
   ```toml
   OPENROUTER_API_KEY = "sk-or-your-key-here"
   ```
5. Click **Deploy**
6. Repeat for each task file — each gets its own URL

### Live Links

| Task | Link |
|---|---|
| Task 1 — Tool-Calling Agent | [Launch app](https://huzaifa206-nexe-agent-internship-task1task1-zo9sdh.streamlit.app/) |
| Task 2 — Calculator Agent | [Launch app](https://huzaifa206-nexe-agent-internship-task2task2-gpzu29.streamlit.app/) |
| Task 3 — Multi-Tool Agent | [Launch app](https://huzaifa206-nexe-agent-internship-task3task3-yzfad7.streamlit.app/) |
| Task 4 — RAG Assistant | [Launch app](https://huzaifa206-nexe-agent-internship-task4task4-ln04gp.streamlit.app/) |
| Task 5 — Business Agent | [Launch app](https://huzaifa206-nexe-agent-internship-task5task5-glbviu.streamlit.app/) |
| Task 6 — Multi-Agent System | [Launch app](https://huzaifa206-nexe-agent-internship-task6task6-d89bfb.streamlit.app/) |


---

## 👤 Author

**Huzaifa Ahmed Siddiqui**  
AI Student · Agentic AI Developer Intern @ Nexe-Agent

[![GitHub](https://img.shields.io/badge/GitHub-Huzaifa206-181717?style=flat&logo=github)](https://github.com/Huzaifa206)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-huzaifaahmedsiddiqui48-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/huzaifaahmedsiddiqui48/)
[![Email](https://img.shields.io/badge/Email-huzaifaasiddiqui@gmail.com-EA4335?style=flat&logo=gmail)](mailto:huzaifaasiddiqui@gmail.com)

---

<div align="center">
  <sub>Built with ❤️ during the Nexe-Agent Agentic AI Developer Internship · 2025</sub>
</div>