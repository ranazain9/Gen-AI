from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter,Language
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("RAG/document_loader/dl-curriculum.pdf")
docs=loader.load()
text="""LangChain is a powerful framework used to build applications with Large Language Models (LLMs). 
It helps developers connect language models with external data sources such as text files, PDFs, databases, and APIs.
One important feature of LangChain is document loading. Document loaders allow users to import data from different file formats and prepare it for processing. 
For example, the TextLoader is used to read plain text (.txt) files and convert them into documents that can be split into smaller chunks.
After loading the text, developers often use text splitters to divide large documents into manageable sections. 
These sections can then be converted into vector embeddings using embedding models.

The embeddings are stored in vector databases such as FAISS, Chroma, or Pinecone. 
Later, when a user asks a question, the system retrieves the most relevant text chunks and sends them to the language model to generate accurate answers.
This process is commonly known as Retrieval-Augmented Generation (RAG). 
RAG improves the performance of language models by providing them with relevant external knowledge instead of relying only on their training data.
LangChain makes it easier to build intelligent applications like chatbots, question-answering systems, and document analysis tools using this workflow."""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=PHYTHON,
    chunk_size=100,
    chunk_overlap=0,
)
result=splitter.split_text(text)
print(result[0])