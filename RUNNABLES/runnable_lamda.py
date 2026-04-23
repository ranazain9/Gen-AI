from langchain_core.runnables import RunnableSequence,RunnablePassthrough,RunnableLambda,RunnableParallel
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
parser=StrOutputParser()
genreate_joke=RunnableSequence(propmt,model,parser)
parrallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'split_text':RunnableLambda(lambda x:len(x.split()))
})

final_chain=RunnableSequence(genreate_joke, parrallel_chain)
result=final_chain.invoke({"topic":"AI"})
print(result)