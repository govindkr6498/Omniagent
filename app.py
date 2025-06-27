import logging
import streamlit as st
from sales_rag_agent import SalesRAGAgent

# Set up logging for Streamlit app
logger = logging.getLogger("streamlit_app")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Streamlit UI
st.set_page_config(page_title="Agentic Sales RAG Bot", layout="centered")
st.title("Agentic Sales RAG Bot")
st.write("A modular sales assistant for lead generation, meeting scheduling, and PDF-based Q&A.")

if 'agent' not in st.session_state:
    st.session_state.agent = SalesRAGAgent(pdf_path='/home/ubuntu/Omniagent/Emaar_FAQ.pdf')
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submitted = st.form_submit_button("Send")
    if submitted and user_input.strip():
        logger.info(f"User input: {user_input}")
        result = st.session_state.agent.process(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", result['response']))
    elif submitted:
        st.warning("Please enter a message.")

# Display chat history
for speaker, msg in st.session_state.chat_history[-20:]:
    if speaker == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

# Show lead info and state for debugging
with st.expander("Debug Info (Lead State)"):
    st.json({
        "lead_info": st.session_state.agent.lead_tool.partial_lead_info,
        "lead_state": st.session_state.agent.lead_tool.state.value
    })
