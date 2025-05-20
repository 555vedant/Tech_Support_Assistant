

import streamlit as st

# lazy-load ChatBot to minimize import issues
@st.cache_resource
def init_chatbot():
    try:
        from RAG_ChatBot import ChatBot
        return ChatBot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

# set page title and header
st.title('Technical Support Bot')
st.markdown("Ask about technical issues with TaskFlow (e.g., 'I am having trouble connecting to Wi-Fi').")

# initialize chatbot
bot = init_chatbot()
if bot is None:
    st.stop()

# generate LLM or static response
def generate_response(input_text):
    try:
        response = bot.query(input_text)
        return response
    except Exception as e:
        return f"Sorry, I couldn't generate a response: {str(e)}. Please try again or contact support."


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm TaskFlow Support Bot. How can I help with your technical issue?"}
    ]

# chat msg
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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