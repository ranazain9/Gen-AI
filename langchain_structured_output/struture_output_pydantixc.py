from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=200
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    summary: str = Field(description="A short summary of the review")
    sentiment: str = Field(description="The sentiment of the review")

structured_output = model.with_structured_output(Review, strict=True)

result = structured_output.invoke(
    "You must return ONLY valid JSON.\n"
    "Keys: summary, sentiment.\n"
    "Write a short review of the movie Inception."
)

print(result)