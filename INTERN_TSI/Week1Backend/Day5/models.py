from sqlalchemy import Column, Integer ,String
from Day1_week1Backend.Day5.database import Base

class Employee(Base):
    __tablename__ ="employees"
    
    id = Column(Integer, primary_key = True)
    name = Column(String)
    department = Column(String)
    salary = Column(Integer)
    