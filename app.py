from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("MONGODB ChatBot")
 ## LLM 
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")

## prompt
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system","Think that you are a helpfull assistant and answer to the following questions"),
    #MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
])

# parser
parser = StrOutputParser()
#input
question = st.text_input("Enter your question")
if question:

    chain = prompt | llm | parser

    result = chain.invoke(question)
    st.write(result)
