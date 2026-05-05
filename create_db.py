#load the pdf
#split into chunks
#create embeddings
#store into chroma

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("Bhagavad-gita-As-It-Is.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embeddings_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(
    chunks,
    embeddings_model,
    persist_directory="Gita-db"
)
