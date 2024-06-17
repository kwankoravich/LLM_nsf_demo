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
    page_title="กอช.ยินดีให้บริการครับ",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# API Key configuration
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ กอช.ยินดีให้บริการครับ"}
    ]

# System prompt for the chat engine
SYSTEM_PROMPT = '''You are กอช. Admin and you need to answer the specific question with your context and explain the answer as long as you can however, you need to answer in Thai language only (ภาษาไทยเท่านั้น)'''

@st.cache_resource(show_spinner=False)
def load_data():
    # with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
    with st.spinner(text="โปรดรอสักครู่ตอนนี้ระบบกำลังทำประมวณผล"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        embed_model = GeminiEmbedding()
        llm = Gemini(model='models/gemini-1.5-flash', google_api_key=google_api_key)

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
        with st.spinner("โปรดรอสักครู่...."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
