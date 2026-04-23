from langchain_groq import ChatGroq
import random
class fake_llm:
    def __init__(self):
        print("LMM")
    def predict(self, propmt):
        response_LIST=[
            "pakistan is great country",
            "Artificial intelligence",
            "india is nothing"
        ]
        return{"response:": random.choice(response_LIST)}
    
class fake_propmt:
    def __init__(self,template,input_variable):
       self.template=template
       self.input_variable= input_variable
    
    def format(self, input_dict):
        return self.template.format(**input_dict)
    

class fake_chain:
    def __init__(self,llm,template):
        self.llm=llm
        self.template= template
    def run(self,input_dict):
        final_prompt=self.template.format(input_dict)
        result_final=self.llm.predict(final_prompt)
        return result_final['response']

template=fake_propmt(
    template="write a {short} poem on topic{topic}",
    input_variable=["short","topic"]
)

llm=fake_llm()
chain=fake_chain(llm,template)
print(chain.run({"short":"length", "topic":"pakistan"}))