from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
template = PromptTemplate(
    template="write the name, age and city of the fictional character Harry Potter.\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

propmt=template.format()
result=model.invoke(propmt)
parsed_result = parser.parse(result.content)

print(parsed_result)

