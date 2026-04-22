from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    name:str = Field(description="The name of the movie")
    rating: int = Field(gt=0, le=10, description="The rating of the movie out of 10")
    review: str = Field(description="The review of the movie")

parser = PydanticOutputParser(pydantic_object=Review, strict=True)


template= PromptTemplate(
    template="give the name rating and review of the following movie {topic}.\n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = template|model|parser
result=chain.invoke({"topic":"KGf"})
print(result)
