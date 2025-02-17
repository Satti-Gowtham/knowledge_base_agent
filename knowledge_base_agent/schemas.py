from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class InputSchema(BaseModel):
    func_name: str
    func_input_data: Optional[Dict[str, Any]] = None

class QueryInput(BaseModel):
    query: str
    top_k: int = 2

class StoreInput(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}

class KnowledgeResponse(BaseModel):
    chunk: str
    chunk_start: int
    chunk_end: int
    full_text: str
    metadata: Dict[str, Any]
    source: Optional[str]
    timestamp: str

class QueryResponse(BaseModel):
    status: str
    data: List[KnowledgeResponse]
