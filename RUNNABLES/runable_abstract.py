from abc import ABC, abstractmethod
import random
class runable(ABC):
    @abstractmethod
    def invoke(self):
        pass
class fake_llm(runable):
    def __init__(self):
        print("LMM")
    
    def invoke(self, propmt):
        response_LIST=[
            "pakistan is great country",
            "Artificial intelligence",
            "india is nothing"
        ]
        return{"response": random.choice(response_LIST)}
    
    def predict(self, propmt):
        response_LIST=[
            "pakistan is great country",
            "Artificial intelligence",
            "india is nothing"
        ]
        return {"response": random.choice(response_LIST)}  # ✅ FIXED
    
class fake_propmt(runable):
    def __init__(self,template,input_variable):
       self.template=template
       self.input_variable= input_variable

    def invoke(self,input_dict):    
        return self.template.format(**input_dict)
    
    def format(self, input_dict):
        return self.template.format(**input_dict)

class chainrunable(runable):
   def __init__(self,runable_list):
       self.runable_list= runable_list
       
   def invoke(self,input_data):
    for runable in self.runable_list:
        input_data=runable.invoke(input_data)
    return input_data

class fake_strparser(runable):
    def invoke(self, input_data):
        return input_data["response"]


# propmt=fake_propmt(
#     template="write a {length} peom on peom{topic}",
#     input_variable=["length","topic"]
# )
# parser=fake_strparser()
# llm=fake_llm()
# chain=chainrunable([propmt,llm,parser])
# result=chain.invoke({"length":"short","topic":"pakistan"})
# print(result)



propmt1=fake_propmt(
    template="write a  joke {topic}",
    input_variable=["topic"]
)


propm2t=fake_propmt(
    template="write a  joke {response}",
    input_variable=["response"]
)
llm=fake_llm()
parser=fake_strparser()

chain1=chainrunable([propmt1,llm])

chain2=chainrunable([propm2t,llm])

final_chain=chainrunable([chain1,chain2,parser])
result=final_chain.invoke({"topic":"Ai"})
print(result)