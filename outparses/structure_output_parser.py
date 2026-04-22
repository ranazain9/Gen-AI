from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

schema1 = [ 
            ResponseSchema(name="Fact_1", description="Fact 1 about the topic"),
            ResponseSchema(name="Fact_2", description="Fact 2 about the topic"),
            ResponseSchema(name="Fact_3", description="Fact 3 about the topic")
            ]

parser= StructuredOutputParser.from_response_schemas(schema1)
template= PromptTemplate(
    template="write 3 facts about the topic {topic}.\n{format_instructions}"
    ,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)          

chain= template | model | parser
chain_result=chain.invoke({"topic": "black hole"})
print(chain_result)