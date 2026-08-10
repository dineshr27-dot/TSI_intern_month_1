from sqlalchemy import Column, Integer,Text
from pgvector.sqlalchemy import Vector
from database import Base

class projectchunk(Base):
    __tablename__ = "project_vector"
    
    chunk_id = Column(Integer, primary_key=True)
    
    chunk = Column(Text)
    
    embedding = Column(Vector(384))