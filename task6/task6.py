import streamlit as st
import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 6 — Multi-Agent System",
    page_icon="🤝",
    layout="wide"
)

# ─── OpenRouter client ────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "deepseek/deepseek-v4-flash:free"

# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION LAYER
# Every message between agents is logged here so we can visualize it in the UI
# ═══════════════════════════════════════════════════════════════════════════════

def post_message(sender: str, receiver: str, message_type: str, content: str):
    """Logs a message on the inter-agent communication bus."""
    if "comm_bus" not in st.session_state:
        st.session_state.comm_bus = []
    st.session_state.comm_bus.append({
        "id":        len(st.session_state.comm_bus) + 1,
        "sender":    sender,
        "receiver":  receiver,
        "type":      message_type,   # task | result | error | status
        "content":   content,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
# Each has its own system prompt, tools, and run() function
# ═══════════════════════════════════════════════════════════════════════════════

# ── Agent 1: Research Agent ───────────────────────────────────────────────────
RESEARCH_SYSTEM = """You are a Research Agent. Your ONLY job is to search the web and return factual, well-structured research findings.
When given a research task:
1. Use web_search to gather information
2. Synthesize findings into clear bullet points
3. Always cite sources (URLs)
4. Be concise — max 300 words
Do NOT do analysis, writing, or math. Only research."""

def research_web_search(query: str, max_results: int = 5) -> dict:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return {
            "query":   query,
            "results": [
                {"title": r.get("title",""), "snippet": r.get("body","")[:300], "url": r.get("href","")}
                for r in results
            ]
        }
    except Exception as e:
        return {"error": str(e)}

RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "research_web_search",
            "description": "Searches the web for information on any topic.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "query":       {"type": "string",  "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Number of results (default 5)"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_research_agent(task: str) -> str:
    """Research Agent: searches the web and returns findings."""
    post_message("Orchestrator", "ResearchAgent", "task", task)

    messages = [
        {"role": "system",  "content": RESEARCH_SYSTEM},
        {"role": "user",    "content": task}
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=RESEARCH_TOOLS
        )
        msg = response.choices[0].message
        messages.append({
            "role":    "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None
        })

        if not msg.tool_calls:
            result = msg.content
            post_message("ResearchAgent", "Orchestrator", "result", result[:200] + "...")
            return result

        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            output = research_web_search(**args)
            post_message("ResearchAgent", "ResearchAgent", "status", f"Searched: {args.get('query','')}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output)})


# ── Agent 2: Analyst Agent ────────────────────────────────────────────────────
ANALYST_SYSTEM = """You are an Analyst Agent. Your ONLY job is to analyze data and extract insights.
When given data to analyze:
1. Identify key patterns and trends
2. Perform any calculations needed
3. Highlight strengths, weaknesses, opportunities, threats if relevant
4. Give 3-5 clear, actionable insights
5. Be structured — use numbered lists
Do NOT search the web or write reports. Only analyze."""

ANALYST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "calculate_metric",
            "description": "Calculates a business or statistical metric.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "formula": {"type": "string", "description": "Python math expression"},
                    "values":  {"type": "object", "description": "Variable values"}
                },
                "required": ["formula", "values"]
            }
        }
    }
]

def analyst_calculate(formula: str, values: dict) -> dict:
    try:
        result = eval(formula, {"__builtins__": {}}, values)
        return {"formula": formula, "values": values, "result": round(result, 4)}
    except Exception as e:
        return {"error": str(e)}

def run_analyst_agent(task: str, data: str) -> str:
    """Analyst Agent: analyzes provided data and returns insights."""
    full_task = f"{task}\n\nData to analyze:\n{data}"
    post_message("Orchestrator", "AnalystAgent", "task", task)

    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user",   "content": full_task}
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=ANALYST_TOOLS
        )
        msg = response.choices[0].message
        messages.append({
            "role":    "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None
        })

        if not msg.tool_calls:
            result = msg.content
            post_message("AnalystAgent", "Orchestrator", "result", result[:200] + "...")
            return result

        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            output = analyst_calculate(**args)
            post_message("AnalystAgent", "AnalystAgent", "status", f"Calculated: {args.get('formula','')}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output)})


# ── Agent 3: Writer Agent ─────────────────────────────────────────────────────
WRITER_SYSTEM = """You are a Writer Agent. Your ONLY job is to write clear, professional content.
When given content to write:
1. Structure it with proper headings
2. Use professional but accessible language
3. Make it engaging and well-formatted (use markdown)
4. Include an executive summary at the top
5. End with clear next steps or conclusions
Do NOT search the web or analyze data. Only write."""

def run_writer_agent(task: str, research: str, analysis: str) -> str:
    """Writer Agent: compiles research and analysis into polished content."""
    full_task = f"""{task}

Research findings:
{research}

Analysis & insights:
{analysis}

Write a comprehensive, well-structured report using all the above."""

    post_message("Orchestrator", "WriterAgent", "task", task)

    response = client.chat.completions.create(
        model    = MODEL,
        messages = [
            {"role": "system", "content": WRITER_SYSTEM},
            {"role": "user",   "content": full_task}
        ]
    )
    result = response.choices[0].message.content
    post_message("WriterAgent", "Orchestrator", "result", result[:200] + "...")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR AGENT
# Decides which specialist agents to call and in what order
# ═══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM = """You are an Orchestrator Agent managing a team of specialist agents.
Your team:
- ResearchAgent: searches the web for information
- AnalystAgent:  analyzes data and extracts insights
- WriterAgent:   writes polished reports and content

When given a task:
1. Call delegate_to_research first to gather information
2. Call delegate_to_analyst to analyze the research findings
3. Call delegate_to_writer to produce the final output
4. Return a brief summary of what each agent did

Always delegate — never do the work yourself."""

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "delegate_to_research",
            "description": "Delegates a research task to the Research Agent who will search the web.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "task": {"type": "string", "description": "The research task or question to investigate"}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "delegate_to_analyst",
            "description": "Delegates an analysis task to the Analyst Agent. Provide the data to analyze.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "task": {"type": "string", "description": "What kind of analysis to perform"},
                    "data": {"type": "string", "description": "The data or research findings to analyze"}
                },
                "required": ["task", "data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "delegate_to_writer",
            "description": "Delegates writing to the Writer Agent. Provide research and analysis.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "task":     {"type": "string", "description": "The writing task and format required"},
                    "research": {"type": "string", "description": "Research findings from ResearchAgent"},
                    "analysis": {"type": "string", "description": "Analysis results from AnalystAgent"}
                },
                "required": ["task", "research", "analysis"]
            }
        }
    }
]

def run_orchestrator(user_request: str) -> tuple[str, list]:
    """
    Orchestrator: receives user request, delegates to specialists, returns final output.
    Returns (final_response, delegation_log)
    """
    post_message("User", "Orchestrator", "task", user_request)
    delegation_log = []

    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user",   "content": user_request}
    ]

    # Store results from each agent to pass between them
    agent_results = {"research": "", "analysis": ""}

    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=ORCHESTRATOR_TOOLS
        )
        msg = response.choices[0].message
        messages.append({
            "role":    "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ] or None
        })

        if not msg.tool_calls:
            post_message("Orchestrator", "User", "result", "Task completed.")
            return msg.content, delegation_log

        for tc in msg.tool_calls:
            args        = json.loads(tc.function.arguments)
            tool_name   = tc.function.name
            agent_output = ""

            if tool_name == "delegate_to_research":
                with st.spinner("🔍 Research Agent working..."):
                    agent_output = run_research_agent(args["task"])
                    agent_results["research"] = agent_output
                delegation_log.append({
                    "agent":  "ResearchAgent",
                    "task":   args["task"],
                    "output": agent_output
                })

            elif tool_name == "delegate_to_analyst":
                with st.spinner("📊 Analyst Agent working..."):
                    agent_output = run_analyst_agent(args["task"], args["data"])
                    agent_results["analysis"] = agent_output
                delegation_log.append({
                    "agent":  "AnalystAgent",
                    "task":   args["task"],
                    "output": agent_output
                })

            elif tool_name == "delegate_to_writer":
                with st.spinner("✍️ Writer Agent working..."):
                    agent_output = run_writer_agent(
                        args["task"],
                        args.get("research", agent_results["research"]),
                        args.get("analysis", agent_results["analysis"])
                    )
                    st.session_state.ma_report = agent_output
                delegation_log.append({
                    "agent":  "WriterAgent",
                    "task":   args["task"],
                    "output": agent_output
                })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      agent_output
            })


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🤝 Task 6 — Multi-Agent System")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
A team of **4 specialized AI agents** collaborate to complete complex tasks:

| Agent | Role |
|---|---|
| 🎯 **Orchestrator** | Receives your request, plans, delegates to specialists |
| 🔍 **Research Agent** | Searches the web for information |
| 📊 **Analyst Agent** | Analyzes data and extracts insights |
| ✍️ **Writer Agent** | Produces polished final output |
""")

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "ma_history"  not in st.session_state: st.session_state.ma_history  = []
if "comm_bus"    not in st.session_state: st.session_state.comm_bus     = []
if "ma_report"   not in st.session_state: st.session_state.ma_report    = None

# ── Sidebar — communication bus ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Agent Communication Bus")
    st.caption("Live messages between agents")

    bus = st.session_state.comm_bus
    if bus:
        # Color map per agent
        colors = {
            "Orchestrator": "🎯",
            "ResearchAgent": "🔍",
            "AnalystAgent":  "📊",
            "WriterAgent":   "✍️",
            "User":          "👤"
        }
        for msg in reversed(bus[-15:]):   # show last 15
            icon_s = colors.get(msg["sender"],   "🤖")
            icon_r = colors.get(msg["receiver"],  "🤖")
            badge  = {"task": "🔵", "result": "🟢", "status": "🟡", "error": "🔴"}.get(msg["type"], "⚪")
            st.markdown(
                f"`{msg['timestamp']}` {badge}  \n"
                f"{icon_s} **{msg['sender']}** → {icon_r} **{msg['receiver']}**  \n"
                f"<small>{msg['content'][:80]}...</small>" if len(msg['content']) > 80
                else f"`{msg['timestamp']}` {badge}  \n{icon_s} **{msg['sender']}** → {icon_r} **{msg['receiver']}**  \n<small>{msg['content']}</small>",
                unsafe_allow_html=True
            )
            st.divider()
    else:
        st.caption("No messages yet. Submit a task to see agents communicate.")

    st.divider()
    if st.button("🔄 Clear all", use_container_width=True):
        for key in ["ma_history","comm_bus","ma_report"]:
            st.session_state[key] = [] if key != "ma_report" else None
        st.rerun()

# ── Suggested tasks ───────────────────────────────────────────────────────────
st.markdown("**💡 Example tasks:**")
c1, c2, c3 = st.columns(3)
examples = [
    "Research and analyze the current state of AI in healthcare",
    "Investigate the Pakistani startup ecosystem and write a market brief",
    "Research remote work trends and write an executive summary",
]
if c1.button(examples[0], use_container_width=True):
    st.session_state.ma_prefill = examples[0]
if c2.button(examples[1], use_container_width=True):
    st.session_state.ma_prefill = examples[1]
if c3.button(examples[2], use_container_width=True):
    st.session_state.ma_prefill = examples[2]

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for entry in st.session_state.ma_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

        if entry.get("delegation_log"):
            for delegation in entry["delegation_log"]:
                agent  = delegation["agent"]
                icons  = {"ResearchAgent": "🔍", "AnalystAgent": "📊", "WriterAgent": "✍️"}
                icon   = icons.get(agent, "🤖")
                with st.expander(f"{icon} {agent} — click to see output"):
                    st.caption(f"**Task:** {delegation['task']}")
                    st.markdown(delegation["output"])

        if entry.get("report"):
            with st.expander("📄 Full Report", expanded=False):
                st.markdown(entry["report"])
                st.download_button(
                    "⬇️ Download",
                    data      = entry["report"],
                    file_name = "multi_agent_report.md",
                    mime      = "text/markdown",
                    key       = f"dl_{entry['timestamp']}"
                )

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("ma_prefill", "")
prompt  = st.chat_input("Give the agent team a task...") or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.ma_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.markdown("🎯 **Orchestrator** received your task. Delegating to specialist agents...")

        try:
            reply, delegation_log = run_orchestrator(prompt)

            st.markdown("---")
            st.markdown(reply)

            # Show each agent's output in expanders
            icons = {"ResearchAgent": "🔍", "AnalystAgent": "📊", "WriterAgent": "✍️"}
            for delegation in delegation_log:
                agent = delegation["agent"]
                icon  = icons.get(agent, "🤖")
                with st.expander(f"{icon} {agent} — click to see output"):
                    st.caption(f"**Task:** {delegation['task']}")
                    st.markdown(delegation["output"])

            # Show report if writer produced one
            report = st.session_state.ma_report
            if report:
                with st.expander("📄 Full Report", expanded=True):
                    st.markdown(report)
                    ts = datetime.now().strftime("%H%M%S")
                    st.download_button(
                        "⬇️ Download Report",
                        data      = report,
                        file_name = "multi_agent_report.md",
                        mime      = "text/markdown",
                        key       = f"dl_new_{ts}"
                    )

            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.ma_history.append({
                "role":           "assistant",
                "content":        reply,
                "delegation_log": delegation_log,
                "report":         report,
                "timestamp":      ts
            })
            st.rerun()

        except Exception as e:
            err = f"❌ Error: {str(e)}"
            st.error(err)
            st.session_state.ma_history.append({
                "role": "assistant", "content": err, "timestamp": datetime.now().strftime("%H%M%S")
            })