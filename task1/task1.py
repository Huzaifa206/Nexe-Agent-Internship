import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Task 1 — Tool-Calling Agent",
    page_icon="🔧",
    layout="centered"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

def get_weather(city: str) -> dict:
    """Returns fake weather data for demo purposes."""
    data = {
        "karachi":   {"temp_c": 34, "condition": "Sunny",  "humidity": "60%"},
        "lahore":    {"temp_c": 29, "condition": "Cloudy", "humidity": "72%"},
        "islamabad": {"temp_c": 25, "condition": "Rainy",  "humidity": "85%"},
        "dubai":     {"temp_c": 38, "condition": "Hot",    "humidity": "55%"},
        "london":    {"temp_c": 14, "condition": "Foggy",  "humidity": "80%"},
    }
    result = data.get(city.lower())
    if not result:
        return {"error": f"City '{city}' not found. Try: Karachi, Lahore, Islamabad, Dubai, London."}
    return result

def calculate(expression: str) -> dict:
    """Safely evaluates a math expression."""
    try:
        allowed = {k: v for k, v in __import__("math").__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"expression": expression, "result": round(result, 6)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}

def get_current_time() -> dict:
    """Returns the current UTC time."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {
        "utc_time": now.strftime("%H:%M:%S"),
        "utc_date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
    }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name e.g. Karachi, Lahore, London"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a math expression and returns the result. Supports +, -, *, /, **, sqrt, log, sin, cos etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression e.g. '100 * 0.18' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current UTC date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

def run_tool(name: str, args: dict) -> str:
    """Calls the right function and returns result as JSON string."""
    if name == "get_weather":
        result = get_weather(**args)
    elif name == "calculate":
        result = calculate(**args)
    elif name == "get_current_time":
        result = get_current_time()
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result)

def run_agent(messages: list) -> tuple[str, list, list]:
    """
    Runs the agentic loop.
    Returns: (final_text, updated_messages, tool_calls_log)
    """
    tool_log = []  # track what tools were called 

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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in (msg.tool_calls or [])
            ] or None
        })

       
        if not msg.tool_calls:
            return msg.content, messages, tool_log

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = run_tool(tc.function.name, args)
            tool_log.append({"tool": tc.function.name, "args": args, "result": json.loads(result)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

st.title("🔧 Task 1 — Tool-Calling Agent")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
This agent can:
- 🌤️ **Check weather** for cities (Karachi, Lahore, Islamabad, Dubai, London)
- 🧮 **Calculate** any math expression (supports `sqrt`, `log`, `**` etc.)
- 🕐 **Tell the current time**

It automatically decides which tool(s) to use based on your message.
""")

st.divider()

st.markdown("**💡 Try these:**")
cols = st.columns(3)
suggestions = [
    "What's the weather in Karachi?",
    "What is 18% of 5000?",
    "What's the weather in London and calculate sqrt(225)?",
]
for i, s in enumerate(suggestions):
    if cols[i].button(s, use_container_width=True):
        st.session_state["prefill"] = s

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Always use tools when the user asks about weather, math, or time. "
                "Be concise and clear in your responses."
            )
        }
    ]
if "display_history" not in st.session_state:
    st.session_state.display_history = []  

for entry in st.session_state.display_history:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("tool_log"):
            with st.expander(f"🔧 {len(entry['tool_log'])} tool call(s) made"):
                for call in entry["tool_log"]:
                    st.markdown(f"**Tool:** `{call['tool']}`")
                    st.markdown(f"**Input:** `{call['args']}`")
                    st.markdown(f"**Result:** `{call['result']}`")
                    st.divider()

prefill = st.session_state.pop("prefill", "")
prompt = st.chat_input("Ask something...", key="chat_input") or prefill

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.display_history.append({"role": "user", "content": prompt})

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, updated, tool_log = run_agent(st.session_state.messages)
                st.session_state.messages = updated

                st.write(reply)

                if tool_log:
                    with st.expander(f"🔧 {len(tool_log)} tool call(s) made"):
                        for call in tool_log:
                            st.markdown(f"**Tool:** `{call['tool']}`")
                            st.markdown(f"**Input:** `{call['args']}`")
                            st.markdown(f"**Result:** `{call['result']}`")
                            st.divider()

                st.session_state.display_history.append({
                    "role": "assistant",
                    "content": reply,
                    "tool_log": tool_log
                })

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.display_history.append({"role": "assistant", "content": err})

if st.session_state.display_history:
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [st.session_state.messages[0]] 
        st.session_state.display_history = []
        st.rerun()