from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qa_pipeline import build_qa_chain, clear_cache
import tempfile
import shutil
import os
import logging
import traceback
import time
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document QA API",
    description="Upload documents and ask questions about their content",
    version="2.0.0"
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Enable CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting up FastAPI application")
    try:
        # Pre-warm the models in background
        from qa_pipeline import initialize_models
        await asyncio.get_event_loop().run_in_executor(executor, initialize_models)
        logger.info("Models pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to pre-load models: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down FastAPI application")
    executor.shutdown(wait=True)
    clear_cache()

def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file."""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    VALID_EXTENSIONS = {".pdf", ".txt", ".docx"}
    
    # Check file extension
    if not file.filename:
        return False, "Filename is required"
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in VALID_EXTENSIONS:
        return False, f"Unsupported file type '{file_ext}'. Use PDF, TXT, or DOCX."
    
    # Check file size
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size == 0:
        return False, "File is empty"
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds 10MB limit"
    
    return True, "Valid"

def validate_question(question: str) -> tuple[bool, str]:
    """Validate the question."""
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    if len(question.strip()) < 3:
        return False, "Question is too short"
    
    if len(question) > 500:
        return False, "Question is too long (max 500 characters)"
    
    return True, "Valid"

async def process_document_question(file_path: str, question: str) -> Dict[str, Any]:
    """Process document and question in a separate thread."""
    loop = asyncio.get_event_loop()
    
    def _process():
        try:
            ask_fn = build_qa_chain(file_path)
            return ask_fn(question)
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            raise
    
    return await loop.run_in_executor(executor, _process)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Document QA API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len(getattr(__import__('qa_pipeline'), 'vectorstore_cache', {}))
    }

@app.post("/upload_and_ask/")
async def upload_and_ask(
    file: UploadFile = File(..., description="Document to analyze (PDF, TXT, or DOCX)"),
    question: str = Form(..., description="Question about the document")
):
    """
    Upload a document and ask a question about its content.
    
    Returns detailed answer with confidence score and source information.
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Validate inputs
        file_valid, file_error = validate_file(file)
        if not file_valid:
            logger.error(f"File validation failed: {file_error}")
            raise HTTPException(status_code=400, detail=file_error)
        
        question_valid, question_error = validate_question(question)
        if not question_valid:
            logger.error(f"Question validation failed: {question_error}")
            raise HTTPException(status_code=400, detail=question_error)
        
        # Save uploaded file
        file_ext = os.path.splitext(file.filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
        
        logger.info(f"File '{file.filename}' saved temporarily. Processing question: '{question[:100]}...'")
        
        # Process document and question
        result = await process_document_question(temp_file_path, question.strip())
        
        # Prepare response
        response = {
            "success": True,
            "answer": result.get("answer", ""),
            "full_answer": result.get("full_answer", ""),
            "confidence": round(result.get("score", 0.0), 3),
            "context": result.get("context", "")[:1000],  # Limit context in response
            "question": question.strip(),
            "filename": file.filename,
            "processing_time": round(time.time() - start_time, 2),
            "source_chunks": len(result.get("source_chunks", [])),
            "metadata": {
                "model_used": "deepset/roberta-base-squad2",
                "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"
            }
        }
        
        # Log performance metrics
        logger.info(f"Successfully processed '{file.filename}' in {response['processing_time']}s. "
                   f"Confidence: {response['confidence']}, Answer length: {len(response['answer'])}")
        
        return response
        
    except HTTPException:
        raise
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Document processing error: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request. Please try again."
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {str(e)}")

@app.post("/clear_cache/")
async def clear_model_cache():
    """Clear the vector store cache to free memory."""
    try:
        clear_cache()
        return {"message": "Cache cleared successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "detail": "An unexpected error occurred",
            "error_type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )