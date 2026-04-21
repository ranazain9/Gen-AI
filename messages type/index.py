from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="hello, how are you?"),
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)
