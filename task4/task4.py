import streamlit as st
import os
import json
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Task 4 — RAG Assistant",
    page_icon="📚",
    layout="centered"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# lazy imports 
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Setting up vector store...")
def load_chroma():
    import chromadb
    chroma_client = chromadb.Client()
    collection    = chroma_client.get_or_create_collection(
        name="rag_docs",
        metadata={"hnsw:space": "cosine"}
    )
    return chroma_client, collection

# document parsing 
def parse_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    try:
        import pypdf, io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text   = "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        return text.strip()
    except Exception as e:
        return f"Error parsing PDF: {e}"

def parse_txt(file_bytes: bytes) -> str:
    """Decodes a plain text file."""
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception:
        return file_bytes.decode("latin-1").strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping chunks for better retrieval.
    overlap ensures context isn't lost at chunk boundaries.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

# vector store operations 
def add_document(filename: str, text: str) -> dict:
    """Chunks a document and stores embeddings in ChromaDB."""
    _, collection = load_chroma()
    embedder      = load_embedder()

    chunks     = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()

    ids       = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]

    try:
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = chunks,
        metadatas  = metadatas
    )
    return {
        "success":  True,
        "filename": filename,
        "chunks":   len(chunks),
        "message":  f"Indexed {len(chunks)} chunks from '{filename}'"
    }

def retrieve_context(query: str, n_results: int = 4) -> dict:
    """Finds the most relevant chunks for a query using cosine similarity."""
    _, collection = load_chroma()
    embedder      = load_embedder()

    if collection.count() == 0:
        return {"error": "No documents indexed yet. Please upload a document first."}

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings = query_embedding,
        n_results        = min(n_results, collection.count()),
        include          = ["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text":       doc,
            "source":     meta["source"],
            "chunk":      meta["chunk"],
            "similarity": round(1 - dist, 3)   
        })

    return {
        "query":   query,
        "chunks":  chunks,
        "context": "\n\n---\n\n".join(c["text"] for c in chunks)
    }

def list_documents() -> dict:
    """Lists all unique documents currently indexed."""
    _, collection = load_chroma()
    if collection.count() == 0:
        return {"documents": [], "total_chunks": 0}

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources  = {}
    for m in all_meta:
        src = m["source"]
        sources[src] = sources.get(src, 0) + 1

    return {
        "documents":    [{"name": k, "chunks": v} for k, v in sources.items()],
        "total_chunks": collection.count()
    }

def delete_document(filename: str) -> dict:
    """Removes all chunks of a document from the vector store."""
    _, collection = load_chroma()
    try:
        existing = collection.get(where={"source": filename})
        if not existing["ids"]:
            return {"error": f"Document '{filename}' not found."}
        collection.delete(ids=existing["ids"])
        return {"success": True, "message": f"Deleted '{filename}' from vector store."}
    except Exception as e:
        return {"error": str(e)}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        "retrieve_context",
            "description": (
                "Searches the vector store for chunks most relevant to the user's question. "
                "Always call this before answering any question about uploaded documents."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query":     {"type": "string",  "description": "The user's question or search query"},
                    "n_results": {"type": "integer", "description": "Number of chunks to retrieve (default 4)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name":        "list_documents",
            "description": "Lists all documents currently indexed in the vector store.",
            "parameters":  {"type": "object", "properties": {}}
        }
    }
]

def run_tool(name: str, args: dict) -> str:
    if name == "retrieve_context":
        return json.dumps(retrieve_context(**args))
    elif name == "list_documents":
        return json.dumps(list_documents())
    return json.dumps({"error": f"Unknown tool: {name}"})

# agentic loop 
def run_agent(messages: list) -> tuple[str, list, list]:
    tool_log = []
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

st.title("📚 Task 4 — RAG Assistant")
st.caption("Agentic AI Developer Internship · Nexe-Agent")

st.markdown("""
Upload any **PDF or TXT** document, then ask questions about it.
The agent retrieves the most relevant sections and answers using only your document content.
""")

st.divider()

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [
        {
            "role":    "system",
            "content": (
                "You are a helpful RAG (Retrieval Augmented Generation) assistant. "
                "When a user asks a question, ALWAYS call retrieve_context first to find relevant document chunks. "
                "Answer ONLY based on the retrieved context — never make things up. "
                "If the context doesn't contain enough info, say so honestly. "
                "Always cite which document and chunk your answer came from. "
                "If no documents are uploaded, tell the user to upload one first."
            )
        }
    ]
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

with st.sidebar:
    st.markdown("### 📁 Document Manager")

    uploaded = st.file_uploader(
        "Upload PDF or TXT",
        type      = ["pdf", "txt"],
        accept_multiple_files = True
    )

    if uploaded:
        for file in uploaded:
            if file.name not in st.session_state.indexed_docs:
                with st.spinner(f"Indexing {file.name}..."):
                    raw  = file.read()
                    text = parse_pdf(raw) if file.name.endswith(".pdf") else parse_txt(raw)
                    if text and not text.startswith("Error"):
                        result = add_document(file.name, text)
                        if result.get("success"):
                            st.session_state.indexed_docs.append(file.name)
                            st.success(f"✅ {file.name} — {result['chunks']} chunks indexed")
                        else:
                            st.error(f"❌ Failed: {result}")
                    else:
                        st.error(f"❌ Could not parse {file.name}")

    st.divider()
    st.markdown("### 📂 Indexed Documents")
    docs_info = list_documents()
    docs      = docs_info.get("documents", [])

    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{doc['name']}**")
            col1.caption(f"{doc['chunks']} chunks")
            if col2.button("🗑️", key=f"del_{doc['name']}"):
                delete_document(doc["name"])
                if doc["name"] in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.remove(doc["name"])
                st.rerun()
    else:
        st.caption("No documents indexed yet.")
        st.caption("Upload a PDF or TXT above to get started.")

    st.divider()
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. **Upload** a document
    2. Text is split into **chunks**
    3. Chunks are **embedded** (converted to vectors)
    4. Your question is also embedded
    5. Most **similar chunks** are retrieved
    6. LLM answers using **only** those chunks
    """)
if not docs:
    st.info("👈 Upload a document in the sidebar to get started.")

for entry in st.session_state.rag_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_log"):
            for call in entry["tool_log"]:
                if call["tool"] == "retrieve_context":
                    chunks = call["result"].get("chunks", [])
                    if chunks:
                        with st.expander(f"📎 {len(chunks)} chunks retrieved"):
                            for i, chunk in enumerate(chunks):
                                st.markdown(f"**Chunk {i+1}** · `{chunk['source']}` · similarity: `{chunk['similarity']}`")
                                st.markdown(f"> {chunk['text'][:300]}...")
                                st.divider()

suggestions = [
    "Summarize the main points of the document",
    "What does the document say about [topic]?",
    "List all documents you have indexed",
]
if docs:
    st.markdown("**💡 Try asking:**")
    c1, c2, c3 = st.columns(3)
    if c1.button(suggestions[0], use_container_width=True):
        st.session_state.rag_prefill = suggestions[0]
    if c2.button(suggestions[1], use_container_width=True):
        st.session_state.rag_prefill = suggestions[1]
    if c3.button(suggestions[2], use_container_width=True):
        st.session_state.rag_prefill = suggestions[2]

prefill = st.session_state.pop("rag_prefill", "")
prompt  = st.chat_input("Ask a question about your documents...") or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.rag_history.append({"role": "user", "content": prompt})
    st.session_state.rag_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                reply, updated, tool_log = run_agent(st.session_state.rag_messages)
                st.session_state.rag_messages = updated

                st.markdown(reply)

                for call in tool_log:
                    if call["tool"] == "retrieve_context":
                        chunks = call["result"].get("chunks", [])
                        if chunks:
                            with st.expander(f"📎 {len(chunks)} chunks retrieved"):
                                for i, chunk in enumerate(chunks):
                                    st.markdown(f"**Chunk {i+1}** · `{chunk['source']}` · similarity: `{chunk['similarity']}`")
                                    st.markdown(f"> {chunk['text'][:300]}...")
                                    st.divider()

                st.session_state.rag_history.append({
                    "role":     "assistant",
                    "content":  reply,
                    "tool_log": tool_log
                })

            except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.error(err)
                st.session_state.rag_history.append({"role": "assistant", "content": err})

if st.session_state.rag_history:
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.rag_messages = [st.session_state.rag_messages[0]]
        st.session_state.rag_history  = []
        st.rerun()