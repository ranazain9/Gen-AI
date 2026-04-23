from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableParallel
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model=ChatGroq(
    model="llama-3.3-70b-versatile"
)
propmt=PromptTemplate(
    template="write a joke on {topic}", 
    input_variables=["topic"]
)


propmt2=PromptTemplate(
    template="explain the joke {text}", 
    input_variables=["text"]
)

parser=StrOutputParser()
chain_joke= RunnableSequence(propmt,model,parser)
chain_parallel=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(propmt2,model,parser)
})
chain=RunnableSequence(chain_joke,chain_parallel)
print(chain.invoke({"topic":"AI"}))
