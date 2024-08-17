from pydantic import BaseModel
from fastapi import UploadFile

class RAGRequest(BaseModel):
    context: UploadFile
    query: str
    class Config:
        arbitrary_types_allowed = True
        
class RAGResponse(BaseModel):
    answer: str