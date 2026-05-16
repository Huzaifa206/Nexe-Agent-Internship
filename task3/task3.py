import streamlit as st
import os
import json
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
st.set_page_config(
    page_title="Task 3 — Multi-Tool Agent",
    page_icon="🛠️",
    layout="centered"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# SQLite
DB_PATH = "agent_data.db"

def init_db():
    """Creates the notes table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT NOT NULL,
            content   TEXT NOT NULL,
            source    TEXT,
            saved_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


def web_search(query: str, max_results: int = 4) -> dict:
    """Searches the web using DuckDuckGo and returns results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return {"error": "No results found.", "query": query}
        cleaned = [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("body", "")[:300],
                "url":     r.get("href", ""),
            }
            for r in results
        ]
        return {"query": query, "results": cleaned, "count": len(cleaned)}
    except Exception as e:
        return {"error": str(e), "query": query}


def save_to_db(title: str, content: str, source: str = "") -> dict:
    """Saves a note or search result to the SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO notes (title, content, source, saved_at) VALUES (?, ?, ?, ?)",
            (title, content, source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        note_id = cur.lastrowid
        conn.close()
        return {"success": True, "id": note_id, "title": title, "message": f"Saved successfully with ID {note_id}"}
    except Exception as e:
        return {"error": str(e)}


def get_all_notes() -> dict:
    """Retrieves all saved notes from the database."""
    try:
        conn  = sqlite3.connect(DB_PATH)
        cur   = conn.cursor()
        cur.execute("SELECT id, title, content, source, saved_at FROM notes ORDER BY id DESC")
        rows  = cur.fetchall()
        conn.close()
        notes = [
            {"id": r[0], "title": r[1], "content": r[2], "source": r[3], "saved_at": r[4]}
            for r in rows
        ]
        return {"notes": notes, "count": len(notes)}
    except Exception as e:
        return {"error": str(e)}


def delete_note(note_id: int) -> dict:
    """Deletes a note from the database by ID."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        conn.commit()
        conn.close()
        return {"success": True, "message": f"Note {note_id} deleted."}
    except Exception as e:
        return {"error": str(e)}


def send_email(to_email: str, subject: str, body: str) -> dict:
    """Sends an email via Gmail SMTP using credentials from Streamlit secrets or .env."""
    try:
        gmail_user     = os.environ.get("GMAIL_ADDRESS", "")
        gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "")

        if not gmail_user or not gmail_password:
            return {
                "error": "Email credentials not configured.",
                "fix": "Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD to your secrets."
            }

        msg = MIMEMultipart()
        msg["From"]    = gmail_user
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, to_email, msg.as_string())

        return {
            "success":  True,
            "to":       to_email,
            "subject":  subject,
            "message":  "Email sent successfully!"
        }
    except Exception as e:
        return {"error": str(e)}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches the web for current information using DuckDuckGo. "
                "Use this when the user asks about news, facts, or anything requiring up-to-date info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 4, max 8)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_db",
            "description": "Saves a note, summary, or search result to the local SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":   {"type": "string", "description": "Short title for the saved note"},
                    "content": {"type": "string", "description": "The main content or summary to save"},
                    "source":  {"type": "string", "description": "URL or source of the information (optional)"}
                },
                "required": ["title", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_notes",
            "description": "Retrieves all notes saved in the database. Use when user asks to see saved notes.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_note",
            "description": "Deletes a saved note by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "integer", "description": "The ID of the note to delete"}
                },
                "required": ["note_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Sends an email to a specified address. "
                "Use when user explicitly asks to send, email, or share something via email."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_email": {"type": "string", "description": "Recipient email address"},
                    "subject":  {"type": "string", "description": "Email subject line"},
                    "body":     {"type": "string", "description": "Full plain-text email body"}
                },
                "required": ["to_email", "subject", "body"]
            }
        }
    }
]

def run_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return json.dumps(web_search(**args))
    elif name == "save_to_db":
        return json.dumps(save_to_db(**args))
    elif name == "get_all_notes":
        return json.dumps(get_all_notes())
    elif name == "delete_note":
        return json.dumps(delete_note(**args))
    elif name == "send_email":
        return json.dumps(send_email(**args))
    return json.dumps({"error": f"Unknown tool: {name}"})

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
            "role":    "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in (msg.tool_calls or [])
            ] or None
        })

        if not msg.tool_calls:
            return msg.content, messages, tool_log

        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = run_tool(tc.function.name, args)
            parsed = json.loads(result)
            tool_log.append({"tool": tc.function.name, "args": args, "result": parsed})
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

st.title("🛠️ Task 3 — Multi-Tool Agent")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
This agent can:
- 🔍 **Search the web** — DuckDuckGo, no API key needed
- 💾 **Save to database** — SQLite, persists across the session
- 📧 **Send emails** — via Gmail SMTP
- 🗂️ **View & delete** saved notes
""")

# Email setup warning
gmail_set = bool(os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD"))
if not gmail_set:
    st.warning(
        "⚠️ Email tool not configured. "
        "Add `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` to your Streamlit secrets to enable it. "
        "Web search and DB tools work without it.",
        icon="📧"
    )

st.divider()

st.markdown("**💡 Try these:**")
c1, c2, c3 = st.columns(3)
suggestions = [
    "Search for latest AI news and save the top result",
    "Search Python tips and email them to test@example.com",
    "Show me all my saved notes",
]
if c1.button(suggestions[0], use_container_width=True):
    st.session_state.t3_prefill = suggestions[0]
if c2.button(suggestions[1], use_container_width=True):
    st.session_state.t3_prefill = suggestions[1]
if c3.button(suggestions[2], use_container_width=True):
    st.session_state.t3_prefill = suggestions[2]

st.divider()

if "t3_messages" not in st.session_state:
    st.session_state.t3_messages = [
        {
            "role": "system",
            "content": (
                "You are a powerful multi-tool AI assistant. You can search the web, "
                "save information to a database, retrieve saved notes, and send emails. "
                "When asked to search AND save, do both in sequence. "
                "When saving, write a clean summarized version of the content. "
                "Always confirm what actions you've taken at the end of your response."
            )
        }
    ]
if "t3_history" not in st.session_state:
    st.session_state.t3_history = []

with st.sidebar:
    st.markdown("### 💾 Saved Notes")
    notes_data = get_all_notes()
    notes      = notes_data.get("notes", [])

    if notes:
        for note in notes:
            with st.expander(f"#{note['id']} · {note['title'][:30]}"):
                st.caption(f"🕐 {note['saved_at']}")
                st.write(note["content"][:200] + "..." if len(note["content"]) > 200 else note["content"])
                if note["source"]:
                    st.caption(f"Source: {note['source'][:50]}")
                if st.button(f"🗑️ Delete #{note['id']}", key=f"del_{note['id']}"):
                    delete_note(note["id"])
                    st.rerun()
    else:
        st.caption("No notes saved yet.")

    st.divider()
    st.markdown("### ⚙️ Email Config")
    if gmail_set:
        st.success(f"✅ Gmail configured")
    else:
        st.error("❌ Gmail not configured")
        with st.expander("How to set up"):
            st.markdown("""
            1. Enable 2FA on your Gmail
            2. Go to Google Account → Security → App Passwords
            3. Generate a password for "Mail"
            4. Add to Streamlit secrets:
            ```
            GMAIL_ADDRESS = "you@gmail.com"
            GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"
            ```
            """)

for entry in st.session_state.t3_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_log"):
            tool_names = [c["tool"] for c in entry["tool_log"]]
            with st.expander(f"🔧 Tools used: {', '.join(tool_names)}"):
                for call in entry["tool_log"]:
                    st.markdown(f"**`{call['tool']}`**")
                    col1, col2 = st.columns(2)
                    col1.markdown("Input")
                    col1.json(call["args"])
                    col2.markdown("Result")
                    col2.json(call["result"])
                    st.divider()

prefill = st.session_state.pop("t3_prefill", "")
prompt  = st.chat_input("Ask me to search, save, or email something...") or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.t3_history.append({"role": "user", "content": prompt})
    st.session_state.t3_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                reply, updated, tool_log = run_agent(st.session_state.t3_messages)
                st.session_state.t3_messages = updated

                st.markdown(reply)

                if tool_log:
                    tool_names = [c["tool"] for c in tool_log]
                    with st.expander(f"🔧 Tools used: {', '.join(tool_names)}"):
                        for call in tool_log:
                            st.markdown(f"**`{call['tool']}`**")
                            col1, col2 = st.columns(2)
                            col1.markdown("Input")
                            col1.json(call["args"])
                            col2.markdown("Result")
                            col2.json(call["result"])
                            st.divider()

                st.session_state.t3_history.append({
                    "role":     "assistant",
                    "content":  reply,
                    "tool_log": tool_log
                })

                # Refresh sidebar if DB was modified
                if any(c["tool"] in ("save_to_db", "delete_note") for c in tool_log):
                    st.rerun()

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.t3_history.append({"role": "assistant", "content": err})

if st.session_state.t3_history:
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.t3_messages = [st.session_state.t3_messages[0]]
        st.session_state.t3_history  = []
        st.rerun()