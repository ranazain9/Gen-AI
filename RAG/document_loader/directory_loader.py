from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path="",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

doc=loader.load()
print(docs)