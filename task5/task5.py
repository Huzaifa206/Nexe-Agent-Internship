import streamlit as st
import os
import json
import sqlite3
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 5 — Business Agent",
    page_icon="🏢",
    layout="wide"
)

# ─── OpenRouter client ────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# ─── SQLite execution log DB ──────────────────────────────────────────────────
DB_PATH = "business_agent_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS execution_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            goal       TEXT NOT NULL,
            step_num   INTEGER,
            step_name  TEXT,
            tool_used  TEXT,
            input      TEXT,
            output     TEXT,
            status     TEXT,
            timestamp  TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_step(session_id, goal, step_num, step_name, tool_used, input_data, output_data, status="success"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO execution_logs
           (session_id, goal, step_num, step_name, tool_used, input, output, status, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id, goal, step_num, step_name, tool_used,
            json.dumps(input_data), json.dumps(output_data),
            status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )
    conn.commit()
    conn.close()

def get_logs(session_id=None):
    conn = sqlite3.connect(DB_PATH)
    if session_id:
        rows = conn.execute(
            "SELECT * FROM execution_logs WHERE session_id=? ORDER BY step_num",
            (session_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM execution_logs ORDER BY id DESC LIMIT 50"
        ).fetchall()
    conn.close()
    cols = ["id","session_id","goal","step_num","step_name","tool_used","input","output","status","timestamp"]
    return [dict(zip(cols, r)) for r in rows]

# ─── Tool functions ───────────────────────────────────────────────────────────

def create_plan(goal: str, context: str = "") -> dict:
    """
    Uses the LLM to break a business goal into 3-6 concrete, executable steps.
    Returns a structured plan the agent will follow.
    """
    prompt = f"""You are a business planning expert.
Break this goal into 3-6 concrete, actionable steps an AI agent can execute.

Goal: {goal}
{f'Context: {context}' if context else ''}

Respond ONLY with valid JSON in this exact format:
{{
  "goal": "{goal}",
  "steps": [
    {{"step": 1, "name": "Step name", "action": "What to do", "tool": "web_search|analyze_data|write_report|calculate_metric"}},
    ...
  ],
  "success_criteria": "How to know the goal is achieved"
}}"""

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 1000,
    )
    raw = response.choices[0].message.content.strip()
    # strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(raw)
        return {"success": True, "plan": plan}
    except Exception:
        return {"success": False, "raw": raw, "error": "Could not parse plan as JSON"}


def web_search(query: str, max_results: int = 4) -> dict:
    """Searches the web for business intelligence and market data."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"error": "No results found", "query": query}
        return {
            "query":   query,
            "results": [
                {"title": r.get("title",""), "snippet": r.get("body","")[:250], "url": r.get("href","")}
                for r in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_data(data: str, analysis_type: str = "summary") -> dict:
    """
    Performs business analysis on provided text data.
    analysis_type: summary | swot | competitors | trends | recommendations
    """
    prompts = {
        "summary":         f"Summarize this business data in 5 bullet points:\n{data}",
        "swot":            f"Perform a SWOT analysis based on this data:\n{data}",
        "competitors":     f"Identify key competitors and their positioning from this data:\n{data}",
        "trends":          f"Identify 3-5 key trends from this data:\n{data}",
        "recommendations": f"Give 3-5 actionable business recommendations based on this data:\n{data}",
    }
    prompt = prompts.get(analysis_type, prompts["summary"])

    response = client.chat.completions.create(
        model      = MODEL,
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 600,
    )
    return {
        "analysis_type": analysis_type,
        "result":        response.choices[0].message.content.strip()
    }


def write_report(title: str, sections: list, goal: str = "") -> dict:
    """
    Generates a structured business report from provided sections.
    sections: list of {"heading": str, "content": str}
    """
    report_md = f"# {title}\n"
    report_md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
    if goal:
        report_md += f"**Goal:** {goal}\n\n---\n\n"
    for sec in sections:
        report_md += f"## {sec['heading']}\n{sec['content']}\n\n"
    return {
        "success": True,
        "title":   title,
        "report":  report_md,
        "length":  len(report_md)
    }


def calculate_metric(formula: str, values: dict) -> dict:
    """
    Calculates a business metric.
    formula: e.g. 'revenue - costs' or 'profit / revenue * 100'
    values:  e.g. {"revenue": 100000, "costs": 60000, "profit": 40000}
    """
    try:
        result = eval(formula, {"__builtins__": {}}, values)
        return {"formula": formula, "values": values, "result": round(result, 4)}
    except Exception as e:
        return {"error": str(e), "formula": formula}


def mark_step_done(step_num: int, step_name: str, result_summary: str) -> dict:
    """Marks a plan step as completed and logs it."""
    session_id = st.session_state.get("b_session_id", "unknown")
    goal       = st.session_state.get("b_current_goal", "")
    log_step(session_id, goal, step_num, step_name, "mark_step_done", {}, {"summary": result_summary})

    if "b_completed_steps" not in st.session_state:
        st.session_state.b_completed_steps = []
    st.session_state.b_completed_steps.append({
        "step":    step_num,
        "name":    step_name,
        "summary": result_summary,
        "time":    datetime.now().strftime("%H:%M:%S")
    })
    return {"success": True, "step": step_num, "name": step_name, "marked": "completed"}


# ─── Tool schemas ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "create_plan",
            "description": "Breaks a business goal into concrete executable steps. Always call this FIRST when given a new goal.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "goal":    {"type": "string", "description": "The business goal to plan"},
                    "context": {"type": "string", "description": "Any additional context (optional)"}
                },
                "required": ["goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "web_search",
            "description": "Searches the web for market data, competitor info, industry trends, or any business intelligence.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (default 4)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "analyze_data",
            "description": "Analyzes business text data. Use after web_search to extract insights.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "data":          {"type": "string", "description": "The text data to analyze"},
                    "analysis_type": {
                        "type":        "string",
                        "enum":        ["summary", "swot", "competitors", "trends", "recommendations"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["data", "analysis_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "write_report",
            "description": "Compiles all findings into a structured business report. Call this as the final step.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "title":    {"type": "string", "description": "Report title"},
                    "goal":     {"type": "string", "description": "The original business goal"},
                    "sections": {
                        "type":  "array",
                        "items": {
                            "type":       "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        },
                        "description": "List of report sections with heading and content"
                    }
                },
                "required": ["title", "sections"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "calculate_metric",
            "description": "Calculates business metrics like profit margin, ROI, growth rate etc.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "formula": {"type": "string", "description": "Python expression e.g. 'profit / revenue * 100'"},
                    "values":  {"type": "object", "description": "Variable values e.g. {\"profit\": 40000, \"revenue\": 100000}"}
                },
                "required": ["formula", "values"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "mark_step_done",
            "description": "Marks a plan step as completed. Call after finishing each step.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "step_num":       {"type": "integer", "description": "The step number (1, 2, 3...)"},
                    "step_name":      {"type": "string",  "description": "Name of the completed step"},
                    "result_summary": {"type": "string",  "description": "Brief summary of what was accomplished"}
                },
                "required": ["step_num", "step_name", "result_summary"]
            }
        }
    }
]

# ─── Tool dispatcher ──────────────────────────────────────────────────────────
def run_tool(name: str, args: dict) -> str:
    result = {}
    if name == "create_plan":
        result = create_plan(**args)
        if result.get("success") and result.get("plan"):
            st.session_state.b_plan = result["plan"]
    elif name == "web_search":
        result = web_search(**args)
    elif name == "analyze_data":
        result = analyze_data(**args)
    elif name == "write_report":
        result = write_report(**args)
        if result.get("success"):
            st.session_state.b_report = result["report"]
    elif name == "calculate_metric":
        result = calculate_metric(**args)
    elif name == "mark_step_done":
        result = mark_step_done(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}

    # log every tool call
    session_id = st.session_state.get("b_session_id", "unknown")
    goal       = st.session_state.get("b_current_goal", "")
    step_num   = args.get("step_num", 0)
    log_step(session_id, goal, step_num, name, name, args, result)

    return json.dumps(result)

# ─── Agentic loop ─────────────────────────────────────────────────────────────
def run_agent(messages: list, progress_bar=None, status_text=None) -> tuple[str, list, list]:
    tool_log  = []
    tool_count = 0

    while True:
        response = client.chat.completions.create(
            model    = MODEL,
            messages = messages,
            tools    = TOOLS,
        )
        msg = response.choices[0].message
        messages.append({
            "role":    "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (msg.tool_calls or [])
            ] or None
        })

        if not msg.tool_calls:
            if progress_bar:
                progress_bar.progress(100)
            return msg.content, messages, tool_log

        for tc in msg.tool_calls:
            args      = json.loads(tc.function.arguments)
            tool_count += 1

            if status_text:
                status_text.markdown(f"⚙️ Running: **`{tc.function.name}`**...")
            if progress_bar:
                progress_bar.progress(min(tool_count * 15, 90))

            result = run_tool(tc.function.name, args)
            parsed = json.loads(result)
            tool_log.append({"tool": tc.function.name, "args": args, "result": parsed})
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🏢 Task 5 — Autonomous Business Agent")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
Give this agent a **business goal**. It will autonomously:
1. **Plan** — break the goal into concrete steps
2. **Execute** — run each step using tools (search, analyze, calculate)
3. **Report** — compile findings into a structured report
4. **Log** — record every action in a database
""")

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "b_messages" not in st.session_state:
    st.session_state.b_messages = [
        {
            "role":    "system",
            "content": (
                "You are an autonomous business intelligence agent. "
                "When given a business goal, you MUST:\n"
                "1. Call create_plan FIRST to break the goal into steps\n"
                "2. Execute EACH step in order using the appropriate tools\n"
                "3. After each step, call mark_step_done with a summary\n"
                "4. After all steps, call write_report to compile findings\n"
                "5. Present the final report clearly to the user\n\n"
                "Be thorough and autonomous — don't ask for permission between steps. "
                "Complete the full plan without stopping."
            )
        }
    ]
if "b_history"         not in st.session_state: st.session_state.b_history         = []
if "b_plan"            not in st.session_state: st.session_state.b_plan             = None
if "b_report"          not in st.session_state: st.session_state.b_report           = None
if "b_completed_steps" not in st.session_state: st.session_state.b_completed_steps  = []
if "b_session_id"      not in st.session_state:
    st.session_state.b_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "b_current_goal"    not in st.session_state: st.session_state.b_current_goal     = ""

# ── Layout: main + sidebar ────────────────────────────────────────────────────
sidebar = st.sidebar

with sidebar:
    st.markdown("### 🗂️ Execution Plan")
    if st.session_state.b_plan:
        plan   = st.session_state.b_plan
        steps  = plan.get("steps", [])
        done   = {s["step"] for s in st.session_state.b_completed_steps}

        st.caption(f"**Goal:** {plan.get('goal','')[:60]}...")
        st.divider()

        for step in steps:
            icon = "✅" if step["step"] in done else "⏳"
            with st.expander(f"{icon} Step {step['step']}: {step['name']}"):
                st.markdown(f"**Action:** {step['action']}")
                st.markdown(f"**Tool:** `{step['tool']}`")
                matching = [s for s in st.session_state.b_completed_steps if s["step"] == step["step"]]
                if matching:
                    st.success(f"Done: {matching[0]['summary'][:100]}")

        st.divider()
        st.caption(f"**Success criteria:**\n{plan.get('success_criteria','')}")
    else:
        st.caption("No plan yet. Submit a goal to start.")

    st.divider()

    # Execution logs viewer
    st.markdown("### 📋 Execution Logs")
    logs = get_logs(st.session_state.b_session_id)
    if logs:
        st.caption(f"{len(logs)} actions logged this session")
        for log in logs[-5:]:   # show last 5
            st.markdown(f"`{log['timestamp'][-8:]}` **{log['tool_used']}**")
            st.caption(f"Status: {log['status']}")
    else:
        st.caption("No logs yet.")

    st.divider()
    st.markdown("### 🔄 New Session")
    if st.button("Start fresh", use_container_width=True):
        for key in ["b_messages","b_history","b_plan","b_report","b_completed_steps","b_current_goal"]:
            del st.session_state[key]
        st.session_state.b_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

# ── Suggested goals ───────────────────────────────────────────────────────────
st.markdown("**💡 Example goals:**")
c1, c2, c3 = st.columns(3)
goals = [
    "Analyze the AI SaaS market and identify top 3 opportunities",
    "Research competitors of a food delivery startup in Pakistan",
    "Create a business plan outline for an online tutoring platform",
]
if c1.button(goals[0], use_container_width=True):
    st.session_state.b_prefill = goals[0]
if c2.button(goals[1], use_container_width=True):
    st.session_state.b_prefill = goals[1]
if c3.button(goals[2], use_container_width=True):
    st.session_state.b_prefill = goals[2]

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for entry in st.session_state.b_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_log"):
            with st.expander(f"🔧 {len(entry['tool_log'])} agent actions"):
                for call in entry["tool_log"]:
                    cols = st.columns([1, 3])
                    cols[0].code(call["tool"])
                    if call["tool"] == "write_report":
                        cols[1].success("Report generated ✅")
                    elif call["tool"] == "mark_step_done":
                        cols[1].success(f"Step {call['args'].get('step_num')} done ✅")
                    else:
                        cols[1].json(call["result"] if len(json.dumps(call["result"])) < 500 else {"preview": str(call["result"])[:300] + "..."})

# ── Inline report display ─────────────────────────────────────────────────────
if st.session_state.b_report and not any(
    "b_report_shown" in e for e in st.session_state.b_history
):
    with st.expander("📄 Full Report", expanded=True):
        st.markdown(st.session_state.b_report)
        st.download_button(
            "⬇️ Download Report",
            data      = st.session_state.b_report,
            file_name = "business_report.md",
            mime      = "text/markdown"
        )

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("b_prefill", "")
prompt  = st.chat_input("Enter a business goal...") or prefill

if prompt:
    st.session_state.b_current_goal = prompt

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.b_history.append({"role": "user", "content": prompt})
    st.session_state.b_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        progress = st.progress(0)
        status   = st.empty()
        status.markdown("🤔 Creating plan...")

        try:
            reply, updated, tool_log = run_agent(
                st.session_state.b_messages,
                progress_bar = progress,
                status_text  = status
            )
            st.session_state.b_messages = updated
            status.empty()
            progress.empty()

            st.markdown(reply)

            if tool_log:
                with st.expander(f"🔧 {len(tool_log)} agent actions"):
                    for call in tool_log:
                        cols = st.columns([1, 3])
                        cols[0].code(call["tool"])
                        if call["tool"] == "write_report":
                            cols[1].success("Report generated ✅")
                        elif call["tool"] == "mark_step_done":
                            cols[1].success(f"Step {call['args'].get('step_num')} done ✅")
                        else:
                            cols[1].json(call["result"] if len(json.dumps(call["result"])) < 500 else {"preview": str(call["result"])[:300] + "..."})

            st.session_state.b_history.append({
                "role":     "assistant",
                "content":  reply,
                "tool_log": tool_log
            })

            # Show report if generated
            if st.session_state.b_report:
                with st.expander("📄 Full Report", expanded=True):
                    st.markdown(st.session_state.b_report)
                    st.download_button(
                        "⬇️ Download Report",
                        data      = st.session_state.b_report,
                        file_name = "business_report.md",
                        mime      = "text/markdown"
                    )

            st.rerun()

        except Exception as e:
            status.empty()
            progress.empty()
            err = f"❌ Error: {str(e)}"
            st.error(err)
            st.session_state.b_history.append({"role": "assistant", "content": err})