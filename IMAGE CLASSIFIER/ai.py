import streamlit as st
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Offline AI Assistant 🤖", layout="centered")
st.title("🤖 Offline AI Assistant (No API Key)")

# Load model once
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

chatbot = load_model()

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
if prompt := st.chat_input("Ask me anything..."):
    # Show user question
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    response = chatbot(prompt, max_length=150, do_sample=True, temperature=0.7, top_p=0.9)
    reply = response[0]["generated_text"][len(prompt):]

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Save reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
