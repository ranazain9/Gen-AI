from langchain_google_genai import ChatGoogleGenerativeAI 
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=200)
result=llm.invoke("What is the capital of France?")
print(result.content)