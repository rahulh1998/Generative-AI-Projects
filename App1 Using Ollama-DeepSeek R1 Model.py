import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


# Streamlit Framework
st.title("First GenAI App Using Ollama using Deepseek-R1 7B")
question = st.text_input("Ask your question ..")

# Initializing LLM 
llm = Ollama(model = 'deepseek-r1:latest')

# Output Parser
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if question:
    response = chain.invoke(question)
    st.write(response)


# Initialize the model
# ollama_model_name = "gemma3:4b"
# embedding_model = "embeddinggemma:300m"

# # Loading the models
# import subprocess
# subprocess.run(["ollama","pull",ollama_model_name])
# subprocess.run(["ollama","pull",embedding_model])

# Initialize the LLM and Embeddings
# llm = OllamaLLM(model=ollama_model_name, temperature=0.1, max_tokens=512)
# embeddings = OllamaEmbeddings(model=embedding_model)

# Get question from user
# question = st.text_input("Enter your question:", key="question_input")

# # Create a prompt template
# prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""You an expert in responding. Give proper understandable responses to the Question: {question}"""
# )
