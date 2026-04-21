from typing import TypedDict

class person(TypedDict):
    name: str
    age: int

p1 = person(name="John", age=30)
print(p1)