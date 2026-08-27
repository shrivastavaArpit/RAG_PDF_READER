from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


# 1. Load document
loader = TextLoader("document loaders/notes.txt")

docs = loader.load()

print("Document loaded")


# 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print(f"Created {len(chunks)} chunks")


# 3. Create embedding model
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

print("Embedding model created")


# 4. Store embeddings in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector database created successfully!")