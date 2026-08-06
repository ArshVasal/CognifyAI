import streamlit as st
from pypdf import PdfReader
import ollama


# APP CONFIGURATION

st.set_page_config(
    page_title="CognifyAI",
    page_icon="🧠",
    layout="wide"
)


# AI CONFIGURATION

MODEL_NAME = "qwen3:1.7b"

def ask_ai(system_prompt, user_prompt):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
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
        st.error(f"Ollama error: {error}")
        return None


# PDF PROCESSING

def extract_text_from_pdf(file):
    """
    Extract text from every readable page of an uploaded PDF.
    """

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            pages.append(
                f"\n--- Page {page_number} ---\n{text}"
            )

    return "\n".join(pages)


def limit_document_size(text, max_characters=5000):
    """
    Prevent extremely large documents from being sent
    directly to the local model.

    Proper RAG would replace this in a future version.
    """

    if len(text) > max_characters:
        return text[:max_characters]

    return text


# AI FEATURES

# AI FEATURES

def generate_summary(document):
    system_prompt = """
You are CognifyAI, an AI study assistant.

Summarize the provided study material briefly.

Rules:
- Use only information from the document.
- Focus on the 5 most important points.
- Keep the answer concise.
- Use bullet points.
- Do not invent information.
"""

    user_prompt = f"""
Summarize this study material:

DOCUMENT:

{document}
"""

    return ask_ai(system_prompt, user_prompt)


def generate_quiz(document, question_count):
    system_prompt = """
You are CognifyAI, an AI study assistant.

Create a short multiple-choice quiz using only the provided document.

For each question provide:
- Question
- Four choices: A, B, C, D
- Correct answer
- Short explanation
"""

    user_prompt = f"""
Create {question_count} multiple-choice questions.

DOCUMENT:

{document}
"""

    return ask_ai(system_prompt, user_prompt)


def answer_question(document, question):
    system_prompt = """
You are CognifyAI, a document question-answering assistant.

Answer using only the provided document.

Keep the answer concise.
If the document does not contain the answer, say so.
"""

    user_prompt = f"""
DOCUMENT:

{document}

QUESTION:

{question}
"""

    return ask_ai(system_prompt, user_prompt)


# SESSION STATE

if "document" not in st.session_state:
    st.session_state.document = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = ""

if "messages" not in st.session_state:
    st.session_state.messages = []


# SIDEBAR

with st.sidebar:

    st.title("🧠 CognifyAI")

    st.caption("Local AI study assistant")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload course material",
        type=["pdf"]
    )

    if uploaded_file:

        with st.spinner("Reading document..."):

            extracted_text = extract_text_from_pdf(
                uploaded_file
            )

            st.session_state.document = limit_document_size(
                extracted_text
            )

        st.success(f"Loaded {uploaded_file.name}")

        st.caption(
            f"{len(st.session_state.document):,} characters processed"
        )

    st.divider()

    st.caption(f"AI Model: {MODEL_NAME}")
    st.caption("Runs locally with Ollama")

    if st.button("Clear session"):

        st.session_state.document = ""
        st.session_state.summary = ""
        st.session_state.quiz = ""
        st.session_state.messages = []

        st.rerun()


# HEADER

st.title("🧠 CognifyAI")

st.subheader(
    "Turn your course material into summaries, quizzes and answers."
)

st.caption(
    "Powered locally by Qwen3 + Ollama"
)

st.divider()


# NO DOCUMENT SCREEN

if not st.session_state.document:

    st.info("Upload a PDF from the sidebar to get started.")

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
        st.subheader("💬 Ask")
        st.write(
            "Ask questions about your uploaded document."
        )

    st.stop()


# MAIN NAVIGATION

summary_tab, quiz_tab, chat_tab = st.tabs(
    [
        "📝 Summary",
        "🎯 Quiz",
        "💬 Ask CognifyAI"
    ]
)


# SUMMARY

with summary_tab:

    st.header("Study Summary")

    st.write(
        "Generate structured notes from your uploaded material."
    )

    if st.button(
        "Generate summary",
        type="primary"
    ):

        with st.spinner(
            "CognifyAI is reading your notes..."
        ):

            result = generate_summary(
                st.session_state.document
            )

        if result:

            st.session_state.summary = result

        else:

            st.error(
                "CognifyAI couldn't reach the local AI model. "
                "Make sure Ollama is running."
            )

    if st.session_state.summary:

        st.markdown(
            st.session_state.summary
        )


# QUIZ

with quiz_tab:

    st.header("Quiz Generator")

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
            "Creating your quiz..."
        ):

            result = generate_quiz(
                st.session_state.document,
                question_count
            )

        if result:

            st.session_state.quiz = result

        else:

            st.error(
                "CognifyAI couldn't reach the local AI model."
            )

    if st.session_state.quiz:

        st.markdown(
            st.session_state.quiz
        )



# DOCUMENT CHAT

with chat_tab:

    st.header("Ask Your Notes")

    st.caption(
        "Answers are generated using the uploaded document."
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

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = answer_question(
                    st.session_state.document,
                    question
                )

            if answer:

                st.markdown(answer)

            else:

                answer = (
                    "I couldn't connect to the local AI model. "
                    "Make sure Ollama is running."
                )

                st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )



# DEBUG / DOCUMENT VIEW

with st.expander("View extracted document text"):

    st.text_area(
        "Extracted text",
        st.session_state.document,
        height=300
    )