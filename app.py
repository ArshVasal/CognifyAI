import streamlit as st

st.set_page_config(page_title="CognifyAI", page_icon="🧠", layout="wide")

st.title("CognifyAI")
st.subheader("Your AI-powered study assistant")

st.write("pload notes, summarize content, generate quizzes, and ask questions.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
else:
    st.info("Upload a PDF to get started.")