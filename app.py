from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

st.title("MONGODB ChatBot")

 ## LLM 
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

## prompt
prompt = ChatPromptTemplate.from_template(
    
    """
   You are a MongoDB notes assistant.

Use ONLY the information present in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not add explanations that are not present in the context.
3. If asked for topics, return only the topic names found in the context.
4. If the answer is not found in the context, say:
   "This information is not available in the notes."
5. Do not infer, expand, or generate additional MongoDB content.


Context:
{context}

    Question: {input}
    """
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

# parser
parser = StrOutputParser()
#input
question = st.text_input("Enter your question")
if question:

    document_chain = create_stuff_documents_chain(llm, prompt)    
    retriever_chain = create_retrieval_chain(retriever,document_chain)
    docs = retriever.invoke(question)

    for i, doc in enumerate(docs):
        st.write(f"\nChunk {i+1}")
        st.write(doc.page_content[:1000])
    result = retriever_chain.invoke({"input":question})

    st.write("CONTEXT SENT TO LLM")
    st.write(result["context"])

    st.write("ANSWER")
    st.write(result["answer"])
