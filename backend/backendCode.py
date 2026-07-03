"""
FastAPI Backend for Advanced RAG System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import logging
import time
import os
from enum import Enum

#immporting python modules for the RAG system

from maincode import (
    AdvancedRAGConfig,
    AdvancedLangChainRAG
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance
rag_instance: Optional[AdvancedLangChainRAG] = None

# ============ Configuration ============

AUTO_INITIALIZE = True #always intialise the rag instance - load the model and the chunks, so ready to chat.
FORCE_REBUILD = False # to rerun document chunking, indexing from scratch
DEFAULT_DOCUMENTS_FOLDER = "./dataset"



# ============ Pydantic Models ============

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    return_sources: bool = Field(default=True, description="Whether to return source documents")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    question: str
    answer: str
    latency: float
    machine_context: List[str] = []
    sources: Optional[List[Dict[str, Any]]] = None


class StatusResponse(BaseModel):
    """Response model for system status"""
    status: str
    is_initialized: bool
    current_machine_context: List[str] = []
    available_machines: List[str] = []
    chunk_count: Optional[int] = None


class RatingRequest(BaseModel):
    """Request model for rating an answer"""
    rating: str = Field(..., pattern="^(right|wrong|maybe)$", description="Rating: 'right' or 'wrong' or 'maybe'")


class InitializeRequest(BaseModel):
    """Request model for initializing the RAG system"""
    documents_folder: str = Field(..., description="Path to documents folder")
    force_rebuild: bool = Field(default=False, description="Force rebuild index")


class MachineContextRequest(BaseModel):
    """Request model for setting machine context"""
    machine_name: List[str] = []
    original_name: Optional[str] = None # to store the original machine name for retrieval based on the original name(for query rephrasing)

class Metrics(BaseModel):
    """Response model for system metrics"""
    total_queries: int
    rated_queries : int
    avg_latency: float
    median_latency: float
    avg_docs_retrieved: float
    answer_correctness_distribution: Dict[str, float]

# ============ Lifespan Management ============

def initialize_rag_system(documents_folder: str,
                          force_rebuild: bool) -> AdvancedLangChainRAG:
        """
        Initialize the RAG system with documents.
        This function can be called at startup or via API.
        
       
        """
        global rag_instance
        
        #adjust these settings as needed
        config = AdvancedRAGConfig(
        USE_HYBRID_SEARCH=True,
        USE_RERANKING=True,
        ENABLE_MONITORING=True,

        USE_8BIT= False,
        USE_4BIT= True, # 4 Bit quantized Qwen3-30B-A3B
        )
        
        rag_instance = AdvancedLangChainRAG(config)
    
      
        rag_instance.run_pipeline_from_result_dicts(
        force_rebuild= force_rebuild,
        documents_folder= documents_folder,
        model_id = config.LLM_MODEL_ID
        )
        

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initialize RAG system on startup if AUTO_INITIALIZE is enabled.
    """
    global rag_instance
    
    logger.info("Starting RAG API Server...")
    
    # Auto-initialize if enabled and documents folder is configured
    if AUTO_INITIALIZE and os.path.exists(DEFAULT_DOCUMENTS_FOLDER):
        try:
            logger.info(f"Auto-initializing with documents from: {DEFAULT_DOCUMENTS_FOLDER}")
            initialize_rag_system(
                documents_folder=DEFAULT_DOCUMENTS_FOLDER,
                force_rebuild= FORCE_REBUILD
                
            )
            logger.info("RAG system auto-initialized successfully")
            logger.info(f"LLM device: {rag_instance.get_llm_device()} | Using GPU: {rag_instance.is_llm_using_gpu()}")
        except Exception as e:
            logger.error(f"Auto-initialization failed: {e}")
            logger.info("You can manually initialize via POST /initialize")
    else:
        logger.info("Auto-initialization disabled or documents folder not found.")
        logger.info("Initialize manually via POST /initialize endpoint")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down RAG API Server...")
    if rag_instance:
        try:
            rag_instance.monitor.generate_and_write_report() # always write report before shutting down
            rag_instance.unload_llm()
            logger.info("LLM unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============ FastAPI App ============

app = FastAPI(
    title="RAG System API",
    description="API for querying technical documentation using RAG with intent routing",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Helper Functions ============

def get_rag_instance() -> AdvancedLangChainRAG:
    """Get the RAG instance or raise an error if not initialized"""
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please initialize first."
        )
    return rag_instance


# ============ API Endpoints ============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "message": "RAG System API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_instance is not None
    }






@app.post("/initialize", tags=["System"])
async def initialize_system(request: InitializeRequest, background_tasks: BackgroundTasks):
    """
    Initialize or reinitialize the RAG system with documents.
    This can take a while for large document sets.
    """
    global rag_instance
    
    # Validate documents folder exists
    if not os.path.exists(request.documents_folder):
        raise HTTPException(
            status_code=400, 
            detail=f"Documents folder not found: {request.documents_folder}"
        )
    
    try:
        logger.info(f"Initializing RAG system with documents from: {request.documents_folder}")
        
        initialize_rag_system(
            documents_folder=request.documents_folder,
            force_rebuild=request.force_rebuild
        )
        
        chunk_count = len(rag_instance.processed_docs) if rag_instance.processed_docs else 0
        
        return {
            "status": "success",
            "message": "RAG system initialized successfully",
            "chunk_count": chunk_count,
            "documents_folder": request.documents_folder
        }
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the RAG system and get a response.
    Supports intent routing and machine context memory.
    """
    rag = get_rag_instance()
    
    try:

        start_time = time.time()
        response = rag._rag_query(question = request.message, return_sources=request.return_sources, machine_context= rag.get_current_machine(), start_time = start_time)
        
        return ChatResponse(
            question= response.get("question", request.message),
            answer= response.get("answer", ""),
            latency=response.get("latency", time.time() - start_time),
            machine_context= rag.get_current_machine(),
            sources=response.get("sources") if request.return_sources else None
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")


@app.post("/chat/rate", tags=["Chat"])
async def rate_answer(request: RatingRequest):
    """Rate the last answer as right or wrong for monitoring"""
    rag = get_rag_instance()
    
    if rag.monitor:
        success = rag.monitor.update_last_entry_correctness(request.rating)
        if success:
            return {"status": "success", "rating": request.rating}
        else:
            raise HTTPException(status_code=400, detail="No answer to rate")
    else:
        raise HTTPException(status_code=400, detail="Monitoring not enabled")


@app.post("/chat/clear", tags=["Chat"])
async def clear_history():
    """Clear chat history and machine context"""
    rag = get_rag_instance()
    rag.clear_history()
    return {"status": "success", "message": "Chat history and context cleared"}


@app.get("/context", tags=["Context"])
async def get_machine_context():
    """Get the current machine context"""
    rag = get_rag_instance()
    return {
        "machine_context": rag.get_current_machine(),
        "has_context": rag.get_current_machine() is not None
    }


@app.post("/machines/set_context", tags=["Context"])
async def set_machine_context(request: MachineContextRequest):
    """Manually set the machine context"""
    rag = get_rag_instance()
    rag.set_current_machine(request.machine_name)
    rag.set_original_machine_name(request.original_name) # store the original machine name for retrieval based on the original name(for query rephrasing)
    return {
        "status": "success",
        "machine_context": request.machine_name
    }


@app.delete("/context", tags=["Context"])
async def clear_machine_context():
    """Clear the machine context (but keep chat history)"""
    rag = get_rag_instance()
    rag.chat_history.current_machine_context = None
    return {"status": "success", "message": "Machine context cleared"}




@app.post("/rebuild", tags=["System"])
async def rebuild_index(documents_folder: str, chunking_strategy: str = "customSection"):
    """Force rebuild the document index"""
    rag = get_rag_instance()
    
    try:
        rag.run_pipeline(
            documents_folder=documents_folder,
            chunking_strategy=chunking_strategy,
            force_rebuild=True
        )
        
        doc_count = len(rag.processed_docs) if rag.processed_docs else 0
        
        return {
            "status": "success",
            "message": "Index rebuilt successfully",
            "document_count": doc_count
        }
        
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")


@app.get("/metrics", response_model = Metrics, tags=["Monitoring"])
async def get_metrics():
    """Get system metrics and performance data"""
    rag = get_rag_instance()
    
    if not rag.monitor:
        raise HTTPException(status_code=400, detail="Monitoring not enabled")
    
    metrics = rag.monitor.generate_and_write_report()
    
    return Metrics (
        total_queries = metrics.get("total_queries", 0),
        rated_queries = metrics.get("rated_queries",0),
        avg_latency = metrics.get("avg_latency", 0),
        median_latency = metrics.get("median_latency", 0),
        avg_docs_retrieved = metrics.get("avg_docs_retrieved", 0),
        answer_correctness_distribution = metrics.get("answer_correctness_distribution", {})

    )


# ============ Run Server ============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )