from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])
chat_history = []
with open(r"D:\GEN AI WITH LANGCHAIN\dynamic chat template\MESSAGE PLACEHOLDER\chathistory.txt") as f:
    chat_history.extend(f.readlines())

prompt = template.invoke({
    "history": chat_history,
    "query": "Where is my refund?"
})

print(prompt)