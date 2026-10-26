
import os
import tempfile
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="AI Placement Preparation Assistant",
    page_icon="🎓",
    layout="wide",
)

# -------------------- CONFIGURATION --------------------

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
RETRIEVAL_K = 3

# -------------------- PROFESSIONAL UI --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827);
        color: #f8fafc;
    }
    .main-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(
            90deg, #38bdf8, #a78bfa, #f472b6, #34d399
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #172033;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
    }

    .stChatInput textarea {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #38bdf8 !important;
        background-color: #172033 !important;
    }

    .stChatInput textarea::placeholder {
        color: #cbd5e1 !important;
        -webkit-text-fill-color: #cbd5e1 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #f1f5f9 !important;
    }

    [data-testid="stExpander"] {
        background-color: #172033 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #f8fafc !important;
    }

    .stAlert p {
        color: #f8fafc !important;
    }

    
/* Chat input: visible text while typing */
.stChatInput textarea {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    caret-color: #38bdf8 !important;
    background-color: #172033 !important;
}

.stChatInput textarea::placeholder {
    color: #cbd5e1 !important;
    -webkit-text-fill-color: #cbd5e1 !important;
    opacity: 1 !important;
}

/* Chat messages */
.stChatMessage {
    color: #f8fafc !important;
}

.stChatMessage p,
.stChatMessage li,
.stChatMessage span {
    color: #f1f5f9;
}

/* Expanders and source citations */
[data-testid="stExpander"] {
    background-color: #172033;
    border: 1px solid #334155;
    border-radius: 10px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #f8fafc !important;
}

/* Status messages */
.stAlert p {
    color: #f8fafc !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🎓 AI Placement Preparation Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    'Prepare smarter with answers grounded in your uploaded study materials.'
    '</div>',
    unsafe_allow_html=True,
)

# -------------------- SESSION STATE --------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption("Upload your placement preparation PDFs.")

    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.caption(
        f"Chunk size: {CHUNK_SIZE} characters | "
        f"Overlap: {CHUNK_OVERLAP} characters"
    )

    process_clicked = st.button(
        "⚡ Process Documents",
        type="primary",
        use_container_width=True,
    )

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        "1. PDF loading\n"
        "2. Text chunking\n"
        "3. Gemini embeddings\n"
        "4. FAISS similarity search\n"
        "5. Gemini answer generation"
    )

# -------------------- PROCESS DOCUMENTS --------------------

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "GOOGLE_API_KEY is missing. Set it in the terminal "
            "and restart Streamlit."
        )
    else:
        documents = []
        successful_files = []
        failed_files = []

        with st.spinner("Loading and splitting PDF documents..."):
            for uploaded_file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf",
                    ) as temporary_file:
                        temporary_file.write(uploaded_file.getvalue())
                        temporary_path = temporary_file.name

                    try:
                        loader = PyPDFLoader(temporary_path)
                        loaded_documents = loader.load()

                        if not loaded_documents or not any(
                            doc.page_content.strip()
                            for doc in loaded_documents
                        ):
                            failed_files.append(
                                f"{uploaded_file.name}: no extractable text"
                            )
                            continue

                        for document in loaded_documents:
                            document.metadata["source"] = uploaded_file.name

                        documents.extend(loaded_documents)
                        successful_files.append(uploaded_file.name)

                    finally:
                        Path(temporary_path).unlink(missing_ok=True)

                except Exception as error:
                    failed_files.append(
                        f"{uploaded_file.name}: "
                        f"{type(error).__name__}: {error}"
                    )

        if not documents:
            st.error("No readable PDF text was found.")
            for message in failed_files:
                st.warning(message)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True,
            )
            chunks = splitter.split_documents(documents)
            chunks = [
                chunk for chunk in chunks
                if chunk.page_content.strip()
            ]

            if not chunks:
                st.error("No usable text chunks were created.")
            else:
                try:
                    with st.spinner(
                        f"Creating embeddings for {len(chunks)} chunks..."
                    ):
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model=EMBEDDING_MODEL,
                        )
                        new_vectorstore = FAISS.from_documents(
                            chunks,
                            embeddings,
                        )

                    # Replace the knowledge base only after successful indexing.
                    st.session_state.vectorstore = new_vectorstore
                    st.session_state.processed_files = successful_files
                    st.session_state.messages = []

                    st.success(
                        f"Indexed {len(chunks)} chunks from "
                        f"{len(successful_files)} PDF(s)."
                    )

                    if failed_files:
                        st.warning("Some PDFs could not be processed:")
                        for message in failed_files:
                            st.write(f"- {message}")

                except Exception as error:
                    st.error(
                        "Index creation failed. Check your API key, "
                        "model availability, network, and quota."
                    )
                    st.caption(
                        f"{type(error).__name__}: {error}"
                    )

# -------------------- KNOWLEDGE BASE STATUS --------------------

if st.session_state.vectorstore is not None:
    st.success("Knowledge base is ready.")
    st.caption(
        "Processed files: "
        + ", ".join(st.session_state.processed_files)
    )

# -------------------- CHAT HISTORY --------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['file']}** — "
                        f"Page {source['page']}"
                    )

# -------------------- QUESTION INPUT --------------------

question = st.chat_input(
    "Ask about Python, Java, SQL, aptitude, HR, or interviews..."
)

if question:
    if st.session_state.vectorstore is None:
        st.warning("Upload and process your preparation PDFs first.")
    elif not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY is missing.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.status("Searching documents and generating an answer...", expanded=True) as progress:
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": RETRIEVAL_K}
                    )
                    progress.write("Searching relevant document chunks...")
                    retrieved_docs = retriever.invoke(question)
                    progress.write(f"Retrieved {len(retrieved_docs)} chunks.")

                    context_parts = []
                    sources = []
                    seen_sources = set()

                    for document in retrieved_docs:
                        source = Path(
                            document.metadata.get("source", "Unknown")
                        ).name

                        # PyPDFLoader normally stores zero-based page numbers.
                        raw_page = document.metadata.get("page")
                        page_number = (
                            int(raw_page) + 1
                            if raw_page is not None
                            else None
                        )

                        citation = (
                            f"[Source: {source}, "
                            f"Page: {page_number or 'Unknown'}]"
                        )

                        context_parts.append(
                            f"{citation}\n{document.page_content}"
                        )

                        source_key = (source, page_number)
                        if source_key not in seen_sources:
                            sources.append(
                                {
                                    "file": source,
                                    "page": page_number or "Unknown",
                                }
                            )
                            seen_sources.add(source_key)

                    context = "\n\n".join(context_parts)
                    
                    prompt = f"""
You are a placement preparation assistant.
Answer the question using the supplied document excerpts.

Rules:
- Identify relevant facts even when the wording differs from the question.
- Combine information from multiple excerpts when useful.
- For a question about RAG components, use any listed components,
  such as retrieval, chunking, reranking, context selection,
  grounded generation, citations, and guardrails.
- Do not invent facts that are not supported by the excerpts.
- Use the exact fallback only when the excerpts contain no relevant
  information for the question.
- Give a concise, structured answer.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

                    llm = ChatGoogleGenerativeAI(
                        model=LLM_MODEL,
                        temperature=0.1,
                    )

                    response = llm.invoke(prompt)
                    content = response.content

                    if isinstance(content, str):
                        answer = content.strip()
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, str):
                                text_parts.append(block)
                            elif isinstance(block, dict):
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if isinstance(text, str):
                                        text_parts.append(text)
                        answer = "\n".join(text_parts).strip()
                    else:
                        answer = ""

                    fallback = (
                        "I could not find this information in the "
                        "uploaded placement preparation material."
                    )

                    # If Gemini returns the fallback despite relevant
                    # RAG terms in the context, ask once with a simpler prompt.
                    context_lower = context.lower()
                    question_lower = question.lower()
                    relevant_terms = (
                        "rag", "retrieval", "chunking", "reranking",
                        "grounded generation", "citations", "guardrails",
                        "vector", "embedding"
                    )

                    context_has_rag_info = any(
                        term in context_lower for term in relevant_terms
                    )
                    is_rag_question = any(
                        term in question_lower
                        for term in ("rag", "retrieval", "components")
                    )

                    if (
                        not answer
                        or fallback.lower() in answer.lower()
                    ) and context_has_rag_info and is_rag_question:
                        retry_prompt = f"""
Answer this question from the excerpts below.
Summarize the relevant information as bullet points.
Do not use the fallback if the excerpts contain a relevant answer.
Do not add unsupported facts.

EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""
                        retry_response = llm.invoke(retry_prompt)
                        retry_content = retry_response.content

                        if isinstance(retry_content, str):
                            retry_answer = retry_content.strip()
                        elif isinstance(retry_content, list):
                            retry_answer = "\n".join(
                                item if isinstance(item, str)
                                else item.get("text", "")
                                for item in retry_content
                                if isinstance(item, str)
                                or (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                    and isinstance(item.get("text", ""), str)
                                )
                            ).strip()
                        else:
                            retry_answer = ""

                        if retry_answer and fallback.lower() not in retry_answer.lower():
                            answer = retry_answer

                    if not answer:
                        answer = (
                            "The model returned no readable text. Please try again."
                        )

                    st.markdown(answer)


                   
                    progress.update(label="Answer ready!", state="complete", expanded=False)

                    if sources:
                        with st.expander("📄 Sources"):
                            for source in sources:
                                st.markdown(
                                    f"- **{source['file']}** — "
                                    f"Page {source['page']}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as error:
                    progress.update(label="Could not complete the answer", state="error", expanded=True)
                    st.error(
                        "The question could not be processed. "
                        "Check Gemini model availability, API quota, "
                        "and your connection."
                    )
                    st.caption(
                        f"{type(error).__name__}: {error}"
                    )

# -------------------- FOOTER --------------------

st.divider()
st.caption(
    "Built with Python • Streamlit • LangChain • Gemini • FAISS")
