from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable,RunnableBranch,RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class Feedback(BaseModel):
    sentiment:Literal['positive', 'negative'] = Field(description="The sentiment of the feedback")


parser= PydanticOutputParser(pydantic_object=Feedback)


template1 = PromptTemplate(
    template="classify the sentiment of the following feedback text into positive, negative. \n {feedback}.\n {format_instruction}",
    input_variables=["feedback"]
    ,partial_variables={"format_instruction": parser.get_format_instructions()}
)


classifier_chain = template1 | model1 | parser

template2 = PromptTemplate(
    template="write a apropriate response to the following negative feedback. \n {feedback}",
    input_variables=["feedback"]
)

merge= RunnableLambda(lambda x: f"Feedback: {x['feedback']}\nSentiment: {x['sentiment']}")

template3 = PromptTemplate(
    template="write a apropriate response to the following positive feedback. \n {feedback}",
    input_variables=["feedback"]
)

parser1= StrOutputParser()

branch_chain = RunnableBranch(
       ( lambda x: x.sentiment == "positive", template3 | model1 | parser1),
        (lambda x: x.sentiment== "negative", template2 | model2 | parser1),
        RunnableLambda(lambda x: "Sorry, I could not classify the sentiment of the feedback.")
)

chain=classifier_chain | branch_chain

classification_result = chain.invoke({"feedback": "The product is really good and I am satisfied with the quality."})

print(classification_result)
print(chain.get_graph().print_ascii())