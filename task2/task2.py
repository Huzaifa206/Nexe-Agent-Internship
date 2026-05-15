import streamlit as st
import os
import json
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Task 2 — AI Calculator Agent",
    page_icon="🧮",
    layout="centered"
)

# ─── OpenRouter client ────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# ─── Tool functions ───────────────────────────────────────────────────────────

def calculate(expression: str) -> dict:
    """Evaluates a math expression. Supports all math module functions."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        allowed["abs"] = abs
        allowed["round"] = round
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {
            "expression": expression,
            "result": result,
            "formatted": f"{result:,}" if isinstance(result, int) else f"{result:,.6f}".rstrip("0").rstrip(".")
        }
    except Exception as e:
        return {"error": str(e), "expression": expression}


def remember(label: str, value: float) -> dict:
    """Saves a value to memory under a label for later use."""
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = {}
    st.session_state.memory_store[label] = value
    return {"saved": label, "value": value, "all_memory": st.session_state.memory_store}


def recall(label: str) -> dict:
    """Retrieves a saved value from memory by label."""
    store = st.session_state.get("memory_store", {})
    if label not in store:
        available = list(store.keys())
        return {"error": f"'{label}' not found in memory.", "available_labels": available}
    return {"label": label, "value": store[label]}


def recall_all() -> dict:
    """Returns all values currently stored in memory."""
    store = st.session_state.get("memory_store", {})
    if not store:
        return {"memory": {}, "message": "Memory is empty."}
    return {"memory": store, "count": len(store)}


def clear_memory() -> dict:
    """Clears all saved values from memory."""
    st.session_state.memory_store = {}
    return {"message": "Memory cleared successfully."}


def solve_steps(problem: str) -> dict:
    """
    Breaks a word problem into structured steps.
    Returns a step-by-step plan the agent should then execute.
    """
    return {
        "problem": problem,
        "instruction": (
            "Break this problem into numbered steps. "
            "For each step, use the calculate tool to compute the value. "
            "Save important intermediate results using the remember tool. "
            "Present a clean structured summary at the end."
        )
    }


# ─── Tool schemas ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluates any math expression. Supports +, -, *, /, **, %, "
                "sqrt(), log(), log2(), log10(), sin(), cos(), tan(), pi, e, "
                "floor(), ceil(), factorial(), abs(), round(). "
                "Always use this for any calculation, no matter how simple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression e.g. '2 ** 10' or 'sqrt(144) + log(100)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Saves a numeric value to memory under a label. Use this to store intermediate results for multi-step problems.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "A short name for the value e.g. 'monthly_salary', 'tax_amount'"},
                    "value": {"type": "number", "description": "The numeric value to save"}
                },
                "required": ["label", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Retrieves a previously saved value from memory by its label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "The label of the saved value to retrieve"}
                },
                "required": ["label"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_all",
            "description": "Returns all values currently stored in memory. Use this to show the user what has been saved.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_memory",
            "description": "Clears all saved values from memory.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "solve_steps",
            "description": "Use this when the user gives a word problem or multi-step problem. Returns a structured plan to solve it step by step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The full problem statement as given by the user"}
                },
                "required": ["problem"]
            }
        }
    }
]

# ─── Tool dispatcher ──────────────────────────────────────────────────────────
def run_tool(name: str, args: dict) -> str:
    if name == "calculate":
        return json.dumps(calculate(**args))
    elif name == "remember":
        return json.dumps(remember(**args))
    elif name == "recall":
        return json.dumps(recall(**args))
    elif name == "recall_all":
        return json.dumps(recall_all())
    elif name == "clear_memory":
        return json.dumps(clear_memory())
    elif name == "solve_steps":
        return json.dumps(solve_steps(**args))
    return json.dumps({"error": f"Unknown tool: {name}"})

# ─── Agentic loop ─────────────────────────────────────────────────────────────
def run_agent(messages: list) -> tuple[str, list, list]:
    tool_log = []

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )

        msg = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (msg.tool_calls or [])
            ] or None
        })

        # No tool calls → done
        if not msg.tool_calls:
            return msg.content, messages, tool_log

        # Run tools
        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = run_tool(tc.function.name, args)
            parsed = json.loads(result)
            tool_log.append({
                "tool":   tc.function.name,
                "args":   args,
                "result": parsed
            })
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🧮 Task 2 — AI Calculator Agent")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
This agent handles:
- 🔢 **Any math** — basic to advanced (trig, log, factorial, etc.)
- 🧠 **Memory** — saves and recalls values across calculations
- 📋 **Structured output** — shows step-by-step working
- 📝 **Word problems** — breaks them into steps automatically
""")

st.divider()

# ── Suggested prompts ─────────────────────────────────────────────────────────
st.markdown("**💡 Try these:**")
c1, c2, c3 = st.columns(3)
suggestions = [
    "What is 15% of 85000?",
    "Save my salary as 150000 and calculate 30% tax on it",
    "A rectangle is 24.5m × 18.3m. Find area and perimeter, save both.",
]
if c1.button(suggestions[0], use_container_width=True):
    st.session_state.prefill = suggestions[0]
if c2.button(suggestions[1], use_container_width=True):
    st.session_state.prefill = suggestions[1]
if c3.button(suggestions[2], use_container_width=True):
    st.session_state.prefill = suggestions[2]

st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "calc_messages" not in st.session_state:
    st.session_state.calc_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise AI calculator assistant. "
                "ALWAYS use the calculate tool for every calculation — never compute mentally. "
                "For multi-step problems, use solve_steps first, then work through each step. "
                "Use remember to save important intermediate values. "
                "Always present results in a clean, structured format with clear labels. "
                "Show your working — list each step clearly before the final answer."
            )
        }
    ]
if "calc_history" not in st.session_state:
    st.session_state.calc_history = []
if "memory_store" not in st.session_state:
    st.session_state.memory_store = {}

# ── Memory sidebar panel ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Memory Store")
    st.caption("Values saved during this session")

    if st.session_state.memory_store:
        for label, value in st.session_state.memory_store.items():
            col1, col2 = st.columns([2, 1])
            col1.markdown(f"`{label}`")
            col2.markdown(f"**{value:,}**" if isinstance(value, int) else f"**{value:,.4f}**")
        st.divider()
        if st.button("🗑️ Clear memory", use_container_width=True):
            st.session_state.memory_store = {}
            st.rerun()
    else:
        st.caption("Nothing saved yet. Ask the agent to remember a value!")

    st.divider()
    st.markdown("### 📖 Supported Functions")
    st.markdown("""
    `sqrt()` `log()` `log2()` `log10()`
    `sin()` `cos()` `tan()`
    `floor()` `ceil()` `round()`
    `factorial()` `abs()`
    `pi` `e`
    `**(power) ` `%(modulo)` 
    """)

# ── Chat display ──────────────────────────────────────────────────────────────
for entry in st.session_state.calc_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_log"):
            with st.expander(f"🔧 {len(entry['tool_log'])} tool call(s) — click to inspect"):
                for call in entry["tool_log"]:
                    cols = st.columns([1, 2])
                    cols[0].markdown(f"**Tool**")
                    cols[0].code(call["tool"])
                    cols[1].markdown(f"**Result**")
                    cols[1].json(call["result"])

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
prompt  = st.chat_input("Enter a math problem or question...") or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.calc_history.append({"role": "user", "content": prompt})
    st.session_state.calc_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            try:
                reply, updated, tool_log = run_agent(st.session_state.calc_messages)
                st.session_state.calc_messages = updated

                st.markdown(reply)

                if tool_log:
                    with st.expander(f"🔧 {len(tool_log)} tool call(s) — click to inspect"):
                        for call in tool_log:
                            cols = st.columns([1, 2])
                            cols[0].markdown(f"**Tool**")
                            cols[0].code(call["tool"])
                            cols[1].markdown(f"**Result**")
                            cols[1].json(call["result"])

                st.session_state.calc_history.append({
                    "role":     "assistant",
                    "content":  reply,
                    "tool_log": tool_log
                })

                # Refresh sidebar if memory changed
                if any(c["tool"] in ("remember", "clear_memory") for c in tool_log):
                    st.rerun()

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.calc_history.append({"role": "assistant", "content": err})

# ── Clear conversation ────────────────────────────────────────────────────────
if st.session_state.calc_history:
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.calc_messages  = [st.session_state.calc_messages[0]]
        st.session_state.calc_history   = []
        st.session_state.memory_store   = {}
        st.rerun()