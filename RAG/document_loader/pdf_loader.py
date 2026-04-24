from langchain_community.document_loaders import PyPDFLoader
loads=PyPDFLoader(r"D:\GEN AI WITH LANGCHAIN\RAG\document_loader\dl-curriculum.pdf")
docs=loads.load()
print(docs)