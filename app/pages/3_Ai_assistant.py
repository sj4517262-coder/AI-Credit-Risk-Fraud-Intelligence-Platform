import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.genai_assistant import generate_credit_explanation, generate_fraud_explanation, answer_question

st.set_page_config(page_title="AI Assistant", page_icon="🧠", layout="wide")
st.title("🧠 GenAI Risk Assistant")
st.caption("Ask questions about your most recent credit or fraud assessment")

case_type = st.radio("Which assessment do you want to discuss?", ["Credit Risk", "Fraud Detection"], horizontal=True)

if case_type == "Credit Risk":
    if 'last_credit_input' not in st.session_state:
        st.warning("No credit risk assessment found. Please run one on the Credit Risk page first.")
        st.stop()
    input_data = st.session_state['last_credit_input']
    generator = generate_credit_explanation
else:
    if 'last_fraud_input' not in st.session_state:
        st.warning("No fraud assessment found. Please run one on the Fraud Detection page first.")
        st.stop()
    input_data = st.session_state['last_fraud_input']
    generator = generate_fraud_explanation

if st.button("Generate Explanation", type="primary"):
    with st.spinner("Gemini is analyzing the result..."):
        explanation_data = generator(input_data)
    st.session_state['explanation_data'] = explanation_data
    st.session_state['chat_history'] = []

if 'explanation_data' in st.session_state:
    st.divider()
    st.markdown("#### Summary")
    st.write(st.session_state['explanation_data']['explanation'])

    st.divider()
    st.markdown("#### Ask a follow up question")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    for role, text in st.session_state['chat_history']:
        with st.chat_message(role):
            st.write(text)

    question = st.chat_input("Ask something about this case...")

    if question:
        st.session_state['chat_history'].append(("user", question))
        with st.spinner("Thinking..."):
            answer = answer_question(question, st.session_state['explanation_data'])
        st.session_state['chat_history'].append(("assistant", answer))
        st.rerun()