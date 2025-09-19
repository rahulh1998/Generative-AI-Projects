# Importing all the essential libraries
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.callbacks import StreamlitCallbackHandler
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st


import subprocess
# subprocess.run(["ollama",'pull',"gemma3:270m"],check=True)
subprocess.run(["ollama",'pull',"embeddinggemma:300m"],check=True)

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")


arxiv_wrap = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrap)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000))

search = DuckDuckGoSearchRun(apiname='search')

st.title("Langchain chat")
st.write("Streamlit callback handler to display thoughts and actions")


if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'assistant', 'content':'Hi Im a chatbot who can search the web'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


# Side bar
st.sidebar.title("API Settings")
api_key = st.sidebar.text_input("Enter your groq api key : ", type="password")


if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message("user").write(prompt)


llm = ChatGroq(api_key=api_key, model= 'llama-3.1-8b-instant', streaming=True)
tools = [search, arxiv, wiki]

# Name of the agent
search_agent = initialize_agent(tools=tools, llm= llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parse_errors = True)

with st.chat_message('assistant'):
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
    st.session_state.messages.append({'role':'assistant','content':response})
    st.write(response)