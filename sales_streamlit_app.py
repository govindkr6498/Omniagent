import streamlit as st
import uuid
from datetime import datetime
import json
import os
from main import SalesRAGAgent  # Changed from SalesRAGBot
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_file' not in st.session_state:
    st.session_state.chat_file = None

def initialize_chatbot():
    """Initialize the chatbot for the current session."""
    if st.session_state.chatbot is None:
        try:
            st.session_state.chatbot = SalesRAGAgent('Emaar_FAQ.pdf') 
            logger.info(f"Chatbot initialized for session {st.session_state.session_id}")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            st.error("Failed to initialize chatbot. Please try again later.")

def save_chat_history():
    """Save chat history to a file."""
    try:
        # Create chat_history directory if it doesn't exist
        os.makedirs("chat_history", exist_ok=True)
        
        if st.session_state.chat_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.chat_file = f"chat_history/chat_{st.session_state.session_id}_{timestamp}.json"
        
        chat_data = {
            "session_id": st.session_state.session_id,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages
        }
        
        with open(st.session_state.chat_file, "w") as f:
            json.dump(chat_data, f, indent=2)
            
        logger.info(f"Chat history updated for session {st.session_state.session_id}")
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def main():
    st.set_page_config(
        page_title="IQBAIBots",
        page_icon="🤖",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.bot {
            background-color: #475063;
        }
        .chat-message .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
        }
        .chat-message .message {
            color: white;
        }
        .stTextInput>div>div>input {
            color: white;
        }
        .lead-info {
            background-color: #1e1e1e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤖 Omniagent Assistant")
    st.markdown("""
        Welcome to our Omniagent Assistant! I can help you learn more about our products and services.
        Feel free to ask any questions!
    """)

    # Initialize chatbot
    initialize_chatbot()

    # Chat container
    chat_container = st.container()

    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
        # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message immediately and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Show spinner while processing response
        if st.session_state.chatbot:
            with st.spinner("🤖 Bot is thinking..."):
                response = st.session_state.chatbot.process(prompt)

            st.session_state.messages.append({"role": "assistant", "content": response['response']})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response['response'])

            save_chat_history()

            # Rerun to update the UI
            # st.rerun()
        else:
            st.error("Chatbot not initialized. Please try again.")

if __name__ == "__main__":
    main()
