from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome.",
    "The capital of Spain is Madrid."
]
query = "What is the capital of France?"
document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)
similarities = cosine_similarity([query_embedding], document_embeddings)[0]
most_similar_index = np.argmax(similarities)
print(most_similar_index)
print(f"Most similar document: {documents[most_similar_index]}")
