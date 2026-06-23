from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

loader = PyPDFLoader('db notes.pdf')
docs = loader.lazy_load()


splitters = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
chunks = splitters.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)
print("Vector database created successfully")
