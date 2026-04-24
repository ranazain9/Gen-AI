from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

model=ChatGroq(
    model="llama-3.3-70b-versatile"
)

loaders=TextLoader(r"D:\GEN AI WITH LANGCHAIN\RAG\document_loader\sample.txt")
docs=loaders.load()

print(docs[0].page_content)

prompt=PromptTemplate(
    template="write the summary of this following langchain para{topic}",
    input_variables=["topic"])

parser=StrOutputParser()

chain= prompt|model|parser
chain_result=chain.invoke({"topic":docs[0].page_content})
print(chain_result)
