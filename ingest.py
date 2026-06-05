from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("Processing documents...")
loader = PyPDFLoader("/home/lavanya/Desktop/Lavanya/random/rag/Software Engineering.pdf")
documents = loader.load()

# Split text cleanly
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Generate vectors and save permanently to local disk
embedding_function = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db")

print(f"✅ Success! Ingested {len(chunks)} chunks into './chroma_db'")
