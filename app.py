import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# SETUP

load_dotenv()

st.set_page_config(
    page_title="CognifyAI",
    page_icon="🧠",
    layout="wide"
)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)


# HELPER FUNCTIONS

def extract_text_from_pdf(file):
    reader = PdfReader(file)

    text = ""

    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text:
            text += f"\n\n--- Page {page_number + 1} ---\n\n"
            text += page_text

    return text


def limit_text(text, max_characters=50000):
    """
    Prevent extremely large PDFs from being sent to the API.
    We can replace this later with proper chunking/RAG.
    """
    if len(text) > max_characters:
        return text[:max_characters]

    return text


def ask_ai(instructions, user_message):
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            instructions=instructions,
            input=user_message
        )

        return response.output_text

    except Exception as error:

        error_message = str(error)

        if "credit_balance_exhausted" in error_message:
            return (
                "⚠️ The cloud AI service is currently unavailable. "
                "Switch to the local AI provider or try again later."
            )

        if "invalid_api_key" in error_message:
            return (
                "⚠️ The AI service could not authenticate. "
                "Please check the API configuration."
            )

        return (
            "⚠️ CognifyAI couldn't complete the request right now. "
            "Please try again."
        )


def generate_summary(document_text):
    instructions = """
You are CognifyAI, an AI study assistant.

Summarize educational material clearly and accurately.

Your summary should:
- identify the major ideas
- explain important concepts simply
- include important definitions
- preserve important formulas or facts
- use headings and bullet points where appropriate
- avoid adding information that is not present in the document
"""

    prompt = f"""
Summarize the following study material.

DOCUMENT:

{document_text}
"""

    return ask_ai(instructions, prompt)


def generate_quiz(document_text, number_of_questions=5):
    instructions = """
You are CognifyAI, an AI study assistant.

Create a useful university-level quiz using ONLY the provided document.

For every question:
1. Give the question
2. Give four options: A, B, C, D
3. Give the correct answer
4. Give a short explanation

Make the questions test understanding rather than simple memorization.
"""

    prompt = f"""
Create {number_of_questions} multiple-choice questions from this document.

DOCUMENT:

{document_text}
"""

    return ask_ai(instructions, prompt)


def answer_document_question(document_text, question):
    instructions = """
You are CognifyAI, an AI study assistant.

Answer questions using the supplied study material.

Rules:
- Base the answer primarily on the supplied document.
- Explain the answer clearly.
- If the answer is not contained in the document, say that the document
  does not provide enough information.
- Do not pretend information exists in the document when it does not.
"""

    prompt = f"""
DOCUMENT:

{document_text}

QUESTION:

{question}
"""

    return ask_ai(instructions, prompt)


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = ""



# HEADER

st.title("🧠 CognifyAI")

st.subheader("Your AI-powered study assistant")

st.write(
    "Upload course material, generate summaries and quizzes, "
    "and ask questions about your notes."
)

st.divider()


# SIDEBAR

with st.sidebar:

    st.header("CognifyAI")

    st.write("Study smarter with your own course material.")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if st.button("Clear Session"):
        st.session_state.messages = []
        st.session_state.document_text = ""
        st.session_state.summary = ""
        st.session_state.quiz = ""

        st.rerun()


# PDF PROCESSING

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):

        extracted_text = extract_text_from_pdf(uploaded_file)

        st.session_state.document_text = limit_text(extracted_text)

    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    st.sidebar.write(
        f"{len(st.session_state.document_text):,} characters extracted"
    )


# NO DOCUMENT

if not st.session_state.document_text:

    st.info("Upload a PDF from the sidebar to begin.")

    st.markdown(
        """
### What CognifyAI can do

**Summarize**
Turn long lecture notes into organized study notes.

**Generate quizzes**
Create practice multiple-choice questions automatically.

**Ask questions**
Ask questions directly about your uploaded material.
"""
    )

    st.stop()


# MAIN APP TABS

summary_tab, quiz_tab, chat_tab, document_tab = st.tabs(
    [
        "📝 Summary",
        "🎯 Quiz",
        "💬 Ask CognifyAI",
        "📄 Document"
    ]
)


# SUMMARY TAB

with summary_tab:

    st.header("AI Summary")

    st.write(
        "Generate organized study notes from your uploaded document."
    )

    if st.button(
        "Generate Summary",
        type="primary"
    ):

        with st.spinner("CognifyAI is summarizing your notes..."):

            st.session_state.summary = generate_summary(
                st.session_state.document_text
            )

    if st.session_state.summary:

        st.markdown(st.session_state.summary)


# QUIZ TAB

with quiz_tab:

    st.header("Quiz Generator")

    number_of_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=15,
        value=5
    )

    if st.button(
        "Generate Quiz",
        type="primary"
    ):

        with st.spinner("Creating your quiz..."):

            st.session_state.quiz = generate_quiz(
                st.session_state.document_text,
                number_of_questions
            )

    if st.session_state.quiz:

        st.markdown(st.session_state.quiz)


# CHAT TAB

with chat_tab:

    st.header("Ask CognifyAI")

    st.caption(
        "Ask questions based on the PDF you uploaded."
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

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

                answer = answer_document_question(
                    st.session_state.document_text,
                    question
                )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# DOCUMENT TAB

with document_tab:

    st.header("Extracted PDF Text")

    st.caption(
        "This is the text CognifyAI currently sees from your PDF."
    )

    st.text_area(
        "Document content",
        st.session_state.document_text,
        height=600
    )