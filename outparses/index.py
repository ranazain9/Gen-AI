from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

# 1st propmt -> detailed report

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)
# 2nd prompt -> short summary    
template2 = PromptTemplate(
    template="write a 5 line  summary of the following text./n {text}",
    input_variables=["text"]
)

prompt1=template1.invoke({"topic": "the impact of climate change on agriculture"})
result=model.invoke(prompt1)

prompt2=template2.invoke({"text": result.content})
summary=model.invoke(prompt2)

print("Summary:\n", summary.content)