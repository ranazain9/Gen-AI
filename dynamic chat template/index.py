from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "explain {topic} "),
])  

prompt = template.invoke({
    "domain": "AI",
    "topic": "what is AI?"
})
print(prompt)