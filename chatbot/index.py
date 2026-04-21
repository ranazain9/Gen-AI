from langchain_huggingface  import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
from sympy import true
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1", 
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
chat_history = [system_message := SystemMessage(content="You are a helpful assistant.")
                ]
while true:
    user_input = input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Model:", response.content)

print(chat_history)
print("Chat ended.")