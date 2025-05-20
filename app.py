# Streamlit Frontend for TaskFlow Technical Support Chatbot
# Dependencies: streamlit, langchain, langchain-core, langchain-huggingface, transformers, langchain_pinecone, pinecone-client, python-dotenv, pandas, numpy, scikit-learn
# Run: streamlit run app.py
# Note: Ensure RAG_ChatBot.py and tech_support_dataset.csv are in the same directory

import streamlit as st

# Lazy-load ChatBot to minimize import issues
@st.cache_resource
def init_chatbot():
    try:
        from RAG_ChatBot import ChatBot
        return ChatBot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

# Set page title and header
st.title('TaskFlow Technical Support Bot')
st.markdown("Ask about technical issues with TaskFlow (e.g., 'I’m having trouble connecting to Wi-Fi').")

# Initialize chatbot
bot = init_chatbot()
if bot is None:
    st.stop()

# Function to generate LLM or static response
def generate_response(input_text):
    try:
        response = bot.query(input_text)
        return response
    except Exception as e:
        return f"Sorry, I couldn't generate a response: {str(e)}. Please try again or contact support."

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm TaskFlow Support Bot. How can I help with your technical issue?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("Enter your technical issue"):
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Fetching response..."):
            response = generate_response(user_input)
            st.markdown(response)
    # Append assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})