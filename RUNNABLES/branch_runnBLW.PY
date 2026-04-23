from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableParallel,RunnableBranch
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(
    model="llama-3.3-70b-versatile"
)
propmt=PromptTemplate(
    template="write a Report on this topic  {topic}", 
    input_variables=["topic"]
)
parser=StrOutputParser()
genreate_report=RunnableSequence(propmt,model,parser)
propmt2=PromptTemplate(
    template="genrate the summary of the given report {report}",
    input_variables=["report"]
)
branch_runnable=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(propmt2,model,parser)),
    RunnablePassthrough()
)
final_chain=RunnableSequence(genreate_report,branch_runnable)
result=final_chain.invoke({"topic":"pakistan"})
print(result)
print(len(result.split()))