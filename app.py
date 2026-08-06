import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader


# APP CONFIGURATION

st.set_page_config(
    page_title="CognifyAI",
    page_icon="🧠",
    layout="wide"
)


# MODEL CONFIGURATION

LLM_MODEL = "qwen3:1.7b"
EMBEDDING_MODEL = "nomic-embed-text"


# AI GENERATION

def ask_ai(system_prompt, user_prompt):
    """
    Send retrieved context and a prompt to the local Qwen model.
    """

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            think=False
        )

        return response["message"]["content"]

    except Exception as error:
        st.error(f"Local AI error: {error}")
        return None


# PDF PROCESSING

def extract_pages_from_pdf(file):
    """
    Extract text while preserving the page number.

    Returning page information is useful because retrieved chunks
    can later be shown as sources.
    """

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages


# CHUNKING

def chunk_document(pages, chunk_size=1000, overlap=150):
    """
    Split PDF text into smaller overlapping chunks.

    chunk_size:
        Maximum number of characters in each chunk.

    overlap:
        Number of characters shared between neighbouring chunks.

    The overlap helps avoid losing information that falls
    directly on a chunk boundary.
    """

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            start += chunk_size - overlap

    return chunks


# EMBEDDINGS

def create_embeddings(chunks):
    """
    Convert all document chunks into numerical vectors.

    Similar pieces of text should have vectors that point
    in similar directions in embedding space.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    try:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts
        )

        return np.array(
            response["embeddings"],
            dtype=np.float32
        )

    except Exception as error:

        st.error(
            f"Embedding error: {error}"
        )

        return None


def embed_query(question):
    """
    Convert the user's question into an embedding using
    the same embedding model as the document chunks.
    """

    try:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question
        )

        return np.array(
            response["embeddings"][0],
            dtype=np.float32
        )

    except Exception as error:

        st.error(
            f"Query embedding error: {error}"
        )

        return None


# VECTOR SIMILARITY

def cosine_similarity(query_vector, document_vectors):
    """
    Measure semantic similarity between the user's question
    and each document chunk.

    Higher score = more semantically similar.
    """

    query_norm = np.linalg.norm(query_vector)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    denominator = (
        document_norms * query_norm
    )

    denominator[
        denominator == 0
    ] = 1e-10

    scores = np.dot(
        document_vectors,
        query_vector
    ) / denominator

    return scores


# RETRIEVAL

def retrieve_relevant_chunks(
    question,
    chunks,
    chunk_embeddings,
    top_k=3
):
    """
    Embed the question and retrieve the document chunks with
    the highest cosine similarity.
    """

    question_embedding = embed_query(
        question
    )

    if question_embedding is None:
        return []

    scores = cosine_similarity(
        question_embedding,
        chunk_embeddings
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "text": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": float(scores[index])
            }
        )

    return results


# RAG QUESTION ANSWERING

def answer_question_with_rag(
    question,
    chunks,
    chunk_embeddings
):
    """
    Complete RAG pipeline:

    Question
        ↓
    Query embedding
        ↓
    Semantic retrieval
        ↓
    Relevant chunks
        ↓
    LLM
        ↓
    Grounded answer
    """

    relevant_chunks = retrieve_relevant_chunks(
        question,
        chunks,
        chunk_embeddings,
        top_k=3
    )

    if not relevant_chunks:

        return None, []

    context_parts = []

    for chunk in relevant_chunks:

        context_parts.append(
            f"""
SOURCE PAGE {chunk["page"]}

{chunk["text"]}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are CognifyAI, a study assistant that answers
questions using retrieved course material.

Rules:

- Answer using ONLY the retrieved context.
- Do not invent facts.
- If the retrieved context does not contain enough
  information, say that you cannot find the answer
  in the uploaded document.
- Keep the answer concise and clear.
- Mention relevant page numbers when useful.
"""

    user_prompt = f"""
RETRIEVED CONTEXT:

{context}

USER QUESTION:

{question}
"""

    answer = ask_ai(
        system_prompt,
        user_prompt
    )

    return answer, relevant_chunks


# SUMMARY

def generate_summary(document_text):
    """
    Summary generation is separate from RAG because summarization
    usually needs broader document context rather than retrieval
    for one specific question.
    """

    max_characters = 5000

    document_text = document_text[
        :max_characters
    ]

    system_prompt = """
You are CognifyAI, an AI study assistant.

Summarize the provided course material.

Rules:

- Use only the document.
- Identify the five most important ideas.
- Keep the response concise.
- Use bullet points.
- Do not invent information.
"""

    user_prompt = f"""
DOCUMENT:

{document_text}

Create a concise study summary.
"""

    return ask_ai(
        system_prompt,
        user_prompt
    )


# QUIZ

def generate_quiz(
    document_text,
    question_count
):
    """
    Generate practice questions from a limited portion
    of the uploaded document.
    """

    document_text = document_text[
        :5000
    ]

    system_prompt = """
You are CognifyAI, an AI study assistant.

Create multiple-choice questions using only the
provided course material.

For every question include:

Question
A. option
B. option
C. option
D. option

Correct Answer: X

Explanation: short explanation

Do not invent information.
"""

    user_prompt = f"""
Create {question_count} multiple-choice questions
using this material:

{document_text}
"""

    return ask_ai(
        system_prompt,
        user_prompt
    )


# SESSION STATE

if "pages" not in st.session_state:
    st.session_state.pages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_embeddings" not in st.session_state:
    st.session_state.chunk_embeddings = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None


# SIDEBAR

with st.sidebar:

    st.title("🧠 CognifyAI")

    st.caption(
        "Local RAG study assistant"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload course material",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if (
            st.session_state.current_file
            != uploaded_file.name
        ):

            with st.spinner(
                "Processing document..."
            ):

                pages = extract_pages_from_pdf(
                    uploaded_file
                )

                document_text = "\n\n".join(
                    page["text"]
                    for page in pages
                )

                chunks = chunk_document(
                    pages
                )

                embeddings = create_embeddings(
                    chunks
                )

                if embeddings is not None:

                    st.session_state.pages = pages

                    st.session_state.document_text = (
                        document_text
                    )

                    st.session_state.chunks = chunks

                    st.session_state.chunk_embeddings = (
                        embeddings
                    )

                    st.session_state.current_file = (
                        uploaded_file.name
                    )

                    st.session_state.summary = ""
                    st.session_state.quiz = ""
                    st.session_state.messages = []

            if (
                st.session_state.chunk_embeddings
                is not None
            ):

                st.success(
                    f"Loaded {uploaded_file.name}"
                )

    if st.session_state.chunks:

        st.caption(
            f"{len(st.session_state.pages)} pages"
        )

        st.caption(
            f"{len(st.session_state.chunks)} chunks"
        )

        st.caption(
            f"{len(st.session_state.document_text):,} characters"
        )

    st.divider()

    st.caption(
        f"LLM: {LLM_MODEL}"
    )

    st.caption(
        f"Embeddings: {EMBEDDING_MODEL}"
    )

    st.caption(
        "Runs locally with Ollama"
    )

    if st.button(
        "Clear session"
    ):

        st.session_state.pages = []
        st.session_state.document_text = ""
        st.session_state.chunks = []

        st.session_state.chunk_embeddings = None

        st.session_state.summary = ""
        st.session_state.quiz = ""
        st.session_state.messages = []

        st.session_state.current_file = None

        st.rerun()


# HEADER

st.title("🧠 CognifyAI")

st.subheader(
    "Turn your course material into summaries, quizzes and answers."
)

st.caption(
    "Local RAG pipeline powered by Qwen3 + Ollama"
)

st.divider()


# EMPTY STATE

if (
    not st.session_state.chunks
    or st.session_state.chunk_embeddings
    is None
):

    st.info(
        "Upload a PDF from the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("📝 Summarize")

        st.write(
            "Turn lecture notes into concise study material."
        )

    with col2:

        st.subheader("🎯 Quiz")

        st.write(
            "Generate practice questions from your notes."
        )

    with col3:

        st.subheader("🔎 Ask")

        st.write(
            "Retrieve relevant material and ask questions."
        )

    st.stop()


# MAIN NAVIGATION

summary_tab, quiz_tab, chat_tab = st.tabs(
    [
        "📝 Summary",
        "🎯 Quiz",
        "🔎 Ask CognifyAI"
    ]
)


# SUMMARY TAB

with summary_tab:

    st.header(
        "Study Summary"
    )

    if st.button(
        "Generate summary",
        type="primary"
    ):

        with st.spinner(
            "Generating summary..."
        ):

            result = generate_summary(
                st.session_state.document_text
            )

        if result:

            st.session_state.summary = (
                result
            )

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# QUIZ TAB

with quiz_tab:

    st.header(
        "Quiz Generator"
    )

    question_count = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button(
        "Generate quiz",
        type="primary"
    ):

        with st.spinner(
            "Creating quiz..."
        ):

            result = generate_quiz(
                st.session_state.document_text,
                question_count
            )

        if result:

            st.session_state.quiz = (
                result
            )

    if st.session_state.quiz:

        st.markdown(
            st.session_state.quiz
        )


# RAG CHAT TAB

with chat_tab:

    st.header(
        "Ask Your Notes"
    )

    st.caption(
        "Questions use embedding-based semantic retrieval."
    )

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about your notes..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching your notes..."
            ):

                answer, sources = (
                    answer_question_with_rag(
                        question,
                        st.session_state.chunks,
                        st.session_state.chunk_embeddings
                    )
                )

            if answer:

                st.markdown(answer)

                with st.expander(
                    "Retrieved sources"
                ):

                    for number, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"""
**Source {number} — Page {source["page"]}**

Similarity score:
`{source["score"]:.3f}`
"""
                        )

                        st.write(
                            source["text"]
                        )

                        st.divider()

            else:

                answer = (
                    "I couldn't generate an answer."
                )

                st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# RAG DETAILS

with st.expander(
    "RAG pipeline details"
):

    st.write(
        f"""
**Pages extracted:** {len(st.session_state.pages)}

**Chunks created:** {len(st.session_state.chunks)}

**Embedding model:** {EMBEDDING_MODEL}

**Generation model:** {LLM_MODEL}

**Retrieval method:** Cosine similarity

**Top chunks retrieved per question:** 3
"""
    )