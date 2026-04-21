from pydantic import BaseModel
from typing import Optional 

class Student(BaseModel):
    name: str
    age: int
    email: Optional[str] = None

new_student = Student({"name":"Alice", "age": 20})
student = Student(**new_student)
print(student)