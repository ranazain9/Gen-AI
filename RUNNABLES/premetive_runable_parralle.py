from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model=ChatGroq(
    model="llama-3.3-70b-versatile"
)

propmt1=PromptTemplate(
    template="generate a tweet on {topic}",
    input_variables=["topic"]

)


propmt2=PromptTemplate(
    template="generate a linkedin post on  {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()
runable_par=RunnableParallel(
    {
    "tweet":RunnableSequence(propmt1,model,parser),
    "linkedin":RunnableSequence(propmt2,model,parser)}
)
result=runable_par.invoke({"topic":"AI"})
print(result)