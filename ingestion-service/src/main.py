from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import logging

from src.kafka_producer import producer_instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ingestion Service", description="API para ingerir documentos al sistema RAG")

class DocumentIngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentIngestResponse(BaseModel):
    doc_id: str
    status: str

@app.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(request: DocumentIngestRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
        
    doc_id = str(uuid.uuid4())
    logger.info(f"Received document ingestion request for assigned ID: {doc_id}")
    
    success = producer_instance.send_document(
        doc_id=doc_id,
        text=request.text,
        metadata=request.metadata or {}
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to queue document for processing")
        
    return DocumentIngestResponse(doc_id=doc_id, status="queued")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
