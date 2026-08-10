from pydantic import BaseModel

class EmployeeCreate(BaseModel):
    id:int
    name: str
    department: str
    salary: int
    
class EmployeeResponse(BaseModel):
    id: int
    name: str
    department: str
    salary: int

class Config:
    from_attributes = True
    
    