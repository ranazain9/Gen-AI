from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Create semantic chunker
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Step 3: Your text
text = """
LangChain is a powerful framework used to build applications with LLMs.
It connects models with external data sources.
Semantic chunking splits text based on meaning instead of fixed size.
This improves retrieval quality in RAG systems.
"""

# Step 4: Split text
chunks = chunker.split_text(text)

print(chunks)