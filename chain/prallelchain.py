from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model1 = ChatHuggingFace(llm=llm)


model2 = ChatGroq(
    model="llama-3.3-70b-versatile",
   
)


template1 = PromptTemplate(
    template="generate a short  and simple notes on the following text. \n {text}",
    input_variables=["text"]
)
template2 = PromptTemplate(
    template="generate a 5 questions and answers based on the following text. \n {text}",
    input_variables=["text"]
)


template3 = PromptTemplate(
    template="merge the following notes and questions and answers into a single text. \n {notes} \n {question}",
    input_variables=["notes", "question"]
)

parser = StrOutputParser()
parrallel_chain = RunnableParallel({
    "notes": template1 | model1 | parser,
    "question": template2 | model2 | parser
})

chain_merge= parrallel_chain| template3 | model1 | parser

result= chain_merge.invoke({"text": "The solar system is made up of the sun and all the objects that orbit around it, including planets, moons, asteroids, comets, and other celestial bodies. The sun is a star that provides light and heat to the solar system. The planets in the solar system include Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Each planet has its own unique characteristics and features. The solar system also contains many moons that orbit around the planets, as well as asteroids and comets that travel through space."})

print(result)