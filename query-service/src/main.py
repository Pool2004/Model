from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from src.query_handler import query_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Query Service", description="API para consultas RAG")

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3

class SourceDocument(BaseModel):
    text: str
    metadata: Dict[str, Any]
    distance: float

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[SourceDocument]

@app.post("/query", response_model=QueryResponse)
async def perform_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
    logger.info(f"Received query: {req.query}")
    
    # 1. Recuperar contexto
    context_docs = query_instance.search_context(req.query, req.n_results)
    
    # 2. Generar respuesta (RAG)
    final_response = query_instance.generate_response(req.query, context_docs)
    
    return QueryResponse(
        query=req.query,
        response=final_response,
        sources=context_docs
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "db_connected": query_instance.collection is not None
    }
