from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import logging
import time
import os
import hashlib
from typing import Dict, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global resources
qa_pipeline = None
embeddings = None
MODEL_NAME = "deepset/roberta-base-squad2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
vectorstore_cache = {}

def initialize_models():
    """Initialize models globally to avoid reloading."""
    global qa_pipeline, embeddings
    try:
        if qa_pipeline is None:
            logger.info(f"Loading QA model: {MODEL_NAME}")
            start_time = time.time()
            qa_pipeline = pipeline(
                "question-answering", 
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                return_multiple_spans=False,
                handle_impossible_answer=True
            )
            logger.info(f"QA model loaded in {time.time() - start_time:.2f} seconds")
            
        if embeddings is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            start_time = time.time()
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
            
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise

def load_document(file_path: str) -> List:
    """Load a document based on its file extension."""
    logger.info(f"Loading document: {file_path}")
    start_time = time.time()
    
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type. Use PDF, TXT, or DOCX.")
            
        docs = loader.load()
        logger.info(f"Document loaded in {time.time() - start_time:.2f} seconds. Pages: {len(docs)}")
        
        # Clean and validate document content
        cleaned_docs = []
        for doc in docs:
            if doc.page_content and doc.page_content.strip():
                # Clean the content
                content = doc.page_content.replace('\n\n', '\n').replace('\t', ' ')
                content = ' '.join(content.split())  # Remove extra whitespace
                if len(content) > 50:  # Only keep substantial content
                    doc.page_content = content
                    cleaned_docs.append(doc)
        
        if not cleaned_docs:
            raise ValueError("Document appears to be empty or contains no readable text")
            
        logger.info(f"Cleaned document contains {len(cleaned_docs)} valid pages")
        return cleaned_docs
        
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise

def get_file_hash(file_path: str) -> str:
    """Generate a hash for the file to use as cache key."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error generating file hash: {str(e)}")
        return str(time.time())  # Fallback to timestamp

def create_optimized_chunks(docs: List, chunk_size: int = 800, chunk_overlap: int = 200) -> List:
    """Create optimized text chunks with better splitting strategy."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    start_time = time.time()
    chunks = splitter.split_documents(docs)
    
    # Filter out very short chunks
    filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
    
    logger.info(f"Created {len(filtered_chunks)} chunks from {len(chunks)} total in {time.time() - start_time:.2f} seconds")
    return filtered_chunks

def build_qa_chain(file_path: str):
    """Build an optimized QA chain for a given document."""
    try:
        # Initialize models
        initialize_models()

        # Check cache
        file_hash = get_file_hash(file_path)
        if file_hash in vectorstore_cache:
            logger.info("Using cached vector store")
            vectorstore = vectorstore_cache[file_hash]
        else:
            # Load and process document
            docs = load_document(file_path)
            chunks = create_optimized_chunks(docs)

            if not chunks:
                raise ValueError("No valid text chunks could be created from the document")

            # Create vector store efficiently
            logger.info("Creating FAISS vector store")
            start_time = time.time()
            
            # Create in batches for memory efficiency
            batch_size = 32
            vectorstore = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    batch_embeddings = embeddings.embed_documents([doc.page_content for doc in batch])
                    vectorstore.add_embeddings(
                        list(zip([doc.page_content for doc in batch], batch_embeddings)),
                        [doc.metadata for doc in batch]
                    )
            
            vectorstore_cache[file_hash] = vectorstore
            logger.info(f"Vector store created in {time.time() - start_time:.2f} seconds")

        # Create retriever with better search parameters
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 6,
                "score_threshold": 0.3
            }
        )

        def ask(question: str) -> Dict:
            """Ask a question and return a detailed answer with metadata."""
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "score": 0.0,
                    "context": "",
                    "full_answer": "Please provide a valid question to get an answer from the document.",
                    "source_chunks": []
                }

            logger.info(f"Processing question: {question}")
            start_time = time.time()
            
            try:
                # Retrieve relevant documents
                relevant_docs = retriever.get_relevant_documents(question)
                
                if not relevant_docs:
                    logger.warning("No relevant documents found")
                    return {
                        "answer": "No relevant information found in the document.",
                        "score": 0.0,
                        "context": "",
                        "full_answer": "No relevant information was found in the provided document to answer your question. Please try rephrasing your question or check if the document contains the information you're looking for.",
                        "source_chunks": []
                    }

                # Prepare context with better handling
                contexts = []
                source_info = []
                
                for i, doc in enumerate(relevant_docs):
                    content = doc.page_content.strip()
                    if content:
                        contexts.append(content)
                        source_info.append({
                            "chunk_id": i,
                            "content": content[:200] + "..." if len(content) > 200 else content,
                            "metadata": doc.metadata
                        })

                # Combine contexts intelligently
                combined_context = " ".join(contexts)
                
                # Limit context length for the QA model (but don't truncate too aggressively)
                max_context_length = 3000
                if len(combined_context) > max_context_length:
                    # Try to keep complete sentences
                    truncated = combined_context[:max_context_length]
                    last_period = truncated.rfind('.')
                    if last_period > max_context_length * 0.8:
                        combined_context = truncated[:last_period + 1]
                    else:
                        combined_context = truncated

                # Get answer from QA model
                qa_result = qa_pipeline(
                    question=question, 
                    context=combined_context,
                    max_answer_len=200,
                    handle_impossible_answer=True
                )
                
                raw_answer = qa_result["answer"].strip()
                confidence = qa_result["score"]
                
                # Create a more natural full answer
                             # Create a more natural full answer
                if confidence > 0.4 and raw_answer and raw_answer.lower() not in ['', 'unknown', 'unanswerable']:
                    # Format the full answer neatly
                    full_answer = (
                        f"✅ **Answer:** {raw_answer}\n\n"
                        f"📚 **Based on the document**, here is a relevant excerpt:\n\n"
                        f"> {contexts[0][:300]}..." if len(contexts[0]) > 300 else f"> {contexts[0]}"
                    )
                elif confidence > 0.2:
                    full_answer = (
                        f"⚠️ The document gives a partial suggestion:\n\n"
                        f"**Possible Answer:** {raw_answer}\n\n"
                        "🔁 Try rephrasing your question for better results or upload a document with more relevant information."
                    )
                else:
                    full_answer = (
                        f"❌ I couldn't confidently find an answer to your question in the document.\n\n"
                        f"📌 Please check that the document contains the relevant information, or try asking a more specific question."
                    )


                logger.info(f"Answer generated in {time.time() - start_time:.2f} seconds with confidence {confidence:.3f}")
                
                return {
                    "answer": raw_answer,
                    "score": float(confidence),
                    "context": combined_context,
                    "full_answer": full_answer,
                    "source_chunks": source_info
                }

            except Exception as e:
                logger.error(f"Error during question answering: {str(e)}")
                return {
                    "answer": "Error processing question.",
                    "score": 0.0,
                    "context": "",
                    "full_answer": f"An error occurred while processing your question: {str(e)}",
                    "source_chunks": []
                }

        return ask

    except Exception as e:
        logger.error(f"Error building QA chain: {str(e)}")
        raise

def clear_cache():
    """Clear the vector store cache."""
    global vectorstore_cache
    vectorstore_cache.clear()
    logger.info("Vector store cache cleared")