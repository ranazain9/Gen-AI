from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

url = "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"

loader = WebBaseLoader(url)
docs = loader.load()

# 🔥 Limit text (important)
text_data = docs[0].page_content[:2000]

prompt = PromptTemplate(
    template="""
Answer the question based ONLY on the context below.

Context:
{text}

Question:
{question}

Answer clearly and concisely.
""",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the model name?",
    "text": text_data
})

print(result)