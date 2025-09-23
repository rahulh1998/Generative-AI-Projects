import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import subprocess

model_name =  "gemma3:4b"

subprocess.run(["ollama", "pull", model_name], check=True)


# Set the page layout to wide
st.set_page_config(layout="wide")

# Langsmith Tracing
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are an expert in answering. Answer the questions in simple to understand manner.'),
        ('user','Questions{questions}')
    ])

# ## Sidebar for prompt template
# prompt = st.sidebar.text_input("Enter your Prompt Here:")

# Streamlit Framework
st.title("First GenAI App Using Ollama using Gemma3:4B")
question = st.text_input("Ask your question ..")


# Loading the LLM Model 
llm = OllamaLLM(model = model_name)

# Output Parser
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if question:
    response = chain.invoke(question)
    st.write(response)