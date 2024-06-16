import streamlit as st
import dotenv, os
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import (
    VectorStoreIndex, 
    ServiceContext, 
    Document, 
    SimpleDirectoryReader
)

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

dotenv.load_dotenv()

# Streamlit configuration
st.set_page_config(
    page_title="Chat with the Streamlit docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# API Key configuration
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyB20fzux92H6rfE0mfN8flMaeewJGp9_-4')

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ กอช.ยินดีให้บริการครับ"}
    ]

# System prompt for the chat engine
SYSTEM_PROMPT = '''You are กอช. Admin and you need to answer the specific question with your context and explain the answer to be concise however, you need to answer in Thai language only (ภาษาไทยเท่านั้น)'''

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        embed_model = GeminiEmbedding()
        llm = Gemini(model='models/gemini-1.5-flash', google_api_key=GOOGLE_API_KEY)

        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, system_prompt=SYSTEM_PROMPT)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

# Initialize the chat engine
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", system_prompt=SYSTEM_PROMPT, memory=memory, verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
