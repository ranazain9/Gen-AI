from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict,Annotated
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=100
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: Annotated[str, "A short summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review"]

structured_output = model.with_structured_output(Review)

result = structured_output.invoke(
    "Return ONLY JSON with keys 'summary' and 'sentiment'. "
    "Write a short review of the movie Inception."
)

print(result)