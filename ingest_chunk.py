from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

loader = PyPDFLoader("/home/lavanya/Desktop/Lavanya/random/rag/Software Engineering.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(documents)

embedding_function = OllamaEmbeddings(model = "nomic-embed-text")
vector_db = Chroma.from_documents(documents = chunks, 
    embedding = embedding_function, 
    persist_directory = "./chroma_db")
print(f"INGEST {len(chunks)} CHUNKS INTO DATABASE SUCCESSFULLY...")


def ask_rag_bot(user_query:str):
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory = "./chroma_db", embedding_function = embedding_function)

    results = vector_db.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    system_prompt = f"""
    You are a professional, helpful assistant. Answer the user's question using ONLY the provided context below.
    If you do not know the answer based on the context, state clearly that the information is not available.
     Do not use external knowledge or invent facts.
    
    CONTEXT:
    {context}
    """

    response = ollama.chat(
        model = "llama3",
        messages = [
            {"role":"system",
            "content": system_prompt},
            {"roe":"user",
            "content": user_query}
        ]
    )

    return response["message"]["content"]

query = "What is the policy for claiming travel expenses?"
print("AI Response:\n", ask_rag_bot(query))