from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

def ask_rag_bot(user_query: str):
    # Connect directly to your existing saved data
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
    # Retrieve relevant document snippets
    results = vector_db.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    system_prompt = f"""
    You are a professional, helpful assistant. Answer the user's question using ONLY the provided context below.
    If you do not know the answer based on the context, state clearly that the information is not available.
    Do not use external knowledge or invent facts.
    
    CONTEXT:
    {context}
    """
    
    # Note the spelling: 'role' is correct here
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query} 
        ]
    )
    return response['message']['content']

# Run a test query
query = "so you are pre trained. because you know the basic knowledge like how to talk and what to rply, what to answer. so you are llm or rag?"
print("\n--- AI Response ---")
print(ask_rag_bot(query))