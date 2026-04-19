from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model = GoogleGenerativeAI()
result = model.invoke("What is the capital of France?")
print(result.content)
