"""Streamlit UI for the unified customer-service runtime."""
from __future__ import annotations

import uuid
import streamlit as st

from runtime import create_runtime

st.set_page_config(
    page_title="Customer Service AI",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Customer Service AI")
st.caption("Real-time AI customer support assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

runtime = create_runtime()

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Session: {st.session_state.session_id[:8]}…")
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

if not st.session_state.messages:
    st.info("Hello! How can I help you today? You can ask about an order, refund, return, billing issue, or another support request.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Type your message…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing your request…"):
            try:
                result = runtime.process_message(
                    prompt,
                    st.session_state.session_id,
                )
                response = result.get("response", "I couldn't process that request.")
            except Exception:
                response = "Sorry, something went wrong while processing your request."

        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
