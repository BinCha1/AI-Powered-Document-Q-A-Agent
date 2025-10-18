"""
AgenticRAG FastAPI Application
A comprehensive RAG system with conversation memory, document upload, and web search fallback.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import tempfile
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Core imports
from dotenv import load_dotenv
load_dotenv()

# LangChain & Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# Memory imports
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# -----------------------------------------------------
# Logging Configuration
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# FastAPI Initialization
# -----------------------------------------------------
app = FastAPI(
    title="AgenticRAG API",
    description="A comprehensive RAG system with conversation memory and document management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Global Variables
# -----------------------------------------------------
vector_store = None
embedding = None
llm = None
parser = None
search_tool = None
memory = None
rag_app = None

# -----------------------------------------------------
# Pydantic Models
# -----------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or query")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class ChatResponse(BaseModel):
    query: str
    answer: str
    source: str
    session_id: Optional[str] = None
    conversation_length: int
    memory_summary: Dict[str, Any]

class UploadResponse(BaseModel):
    message: str
    filename: str
    total_chunks_added: int

class StatusResponse(BaseModel):
    vector_store_status: str
    document_count: int
    sample_documents: List[str]
    memory_status: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str

# -----------------------------------------------------
# Graph State Definition
# -----------------------------------------------------
class GraphState(TypedDict):
    query: str
    docs: Optional[List[Document]]
    answer: Optional[str]
    source: Optional[str]
    use_rag: Optional[bool]
    chat_history: Optional[List[BaseMessage]]

# -----------------------------------------------------
# Node Functions
# -----------------------------------------------------
def retriever_node(state: GraphState):
    """Retrieve relevant documents from vector store with memory context"""
    query = state["query"]
    chat_history = state.get("chat_history", [])

    enhanced_query = query
    if chat_history:
        recent_context = []
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                recent_context.append(f"Previous question: {msg.content}")
            elif isinstance(msg, AIMessage):
                recent_context.append(f"Previous answer: {msg.content[:100]}...")
        if recent_context:
            enhanced_query = f"{query} (Context: {' '.join(recent_context)})"

    results = vector_store.similarity_search_with_score(enhanced_query, k=5)

    max_distance = 1.0
    filtered_docs = [doc for doc, score in results if score < max_distance]

    logger.info(f"Retriever found {len(filtered_docs)} relevant docs from {len(results)} total")

    use_rag = len(filtered_docs) > 0
    return {"docs": filtered_docs, "use_rag": use_rag}

def rag_node(state: GraphState):
    """Answer question using retrieved documents and conversation history"""
    docs = state.get("docs", [])
    query = state["query"]
    chat_history = state.get("chat_history", [])

    if not docs:
        return {"answer": "No relevant documents found.", "source": "None"}

    context = "\n\n".join([d.page_content for d in docs])

    history_context = ""
    if chat_history:
        history_context = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                history_context += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_context += f"Assistant: {msg.content}\n"

    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent assistant with access to conversation history. 
        Use the provided context and conversation history to answer the question.
        If the answer can be found in the context, provide a clear and accurate response.
        If the answer is not present in the context, respond: "Not found in document".
        
        {history_context}
        
        Context from documents:
        {context}

        Current Question: {question}

        Answer:"""
    )

    chain = prompt | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": query,
        "history_context": history_context
    })

    sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))
    sources_str = ", ".join(sources)

    logger.info(f"RAG answered using {len(docs)} documents with memory context")

    return {"answer": answer, "source": f"RAG (Documents: {sources_str})"}

def web_search_node(state: GraphState):
    """Search the web when documents don't have the answer, with memory context"""
    query = state["query"]
    chat_history = state.get("chat_history", [])

    history_context = ""
    if chat_history:
        history_context = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            if isinstance(msg, HumanMessage):
                history_context += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_context += f"Assistant: {msg.content}\n"

    enhanced_query = query
    if history_context:
        enhanced_query = f"{query} (Context: {history_context})"

    logger.info("Executing web search with memory context...")
    result = search_tool.func(enhanced_query)

    return {"answer": result, "source": "Web Search (SerpAPI)"}

def summarizer_node(state: GraphState):
    """Refine and format the final answer"""
    answer = state.get("answer")

    if not answer or answer == "No relevant documents found.":
        return {"answer": answer}

    prompt = ChatPromptTemplate.from_template(
        """Refine and format the following answer to be clear, concise, and professional.
        Keep the key information but improve readability.

        Answer to refine:
        {answer}

        Refined answer:"""
    )

    chain = prompt | llm | parser
    refined = chain.invoke({"answer": answer})

    logger.info("Answer summarized and refined")
    return {"answer": refined}

def memory_node(state: GraphState):
    """Manage conversation memory - save current interaction"""
    query = state["query"]
    answer = state.get("answer", "")

    memory.save_context({"query": query}, {"answer": answer})
    chat_history = memory.chat_memory.messages

    logger.info(f"Memory updated. Total messages in history: {len(chat_history)}")
    return {"chat_history": chat_history}

def route_after_retrieval(state: GraphState) -> Literal["rag", "web_search"]:
    """Route to RAG or web search based on document retrieval"""
    if state.get("use_rag"):
        logger.info("Routing to RAG")
        return "rag"
    else:
        logger.info("Routing to Web Search")
        return "web_search"

# -----------------------------------------------------
# Initialization Functions
# -----------------------------------------------------
def initialize_rag_system():
    """Initialize the RAG system components"""
    global vector_store, embedding, llm, parser, search_tool, memory, rag_app

    try:
        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )

        parser = StrOutputParser()

        serp_api_key = os.getenv("SERPAPI_API_KEY")
        if serp_api_key:
            serp = SerpAPIWrapper(serpapi_api_key=serp_api_key)
            search_tool = Tool(
                name="Search",
                func=serp.run,
                description="Search web queries when answer not found in documents"
            )
        else:
            logger.warning("SERPAPI_API_KEY not found. Web search will be disabled.")
            search_tool = None

        memory = ConversationBufferWindowMemory(
            k=6,
            memory_key="chat_history",
            return_messages=True,
            input_key="query",
            output_key="answer"
        )

        persist_directory = "chroma_db"
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
            )
            logger.info("Loaded existing vector store")
        else:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding
            )
            logger.info("Created new empty vector store")

        graph = StateGraph(GraphState)
        graph.add_node("retriever", retriever_node)
        graph.add_node("rag", rag_node)
        graph.add_node("web_search", web_search_node)
        graph.add_node("summarizer", summarizer_node)
        graph.add_node("memory", memory_node)

        graph.set_entry_point("retriever")

        graph.add_conditional_edges(
            "retriever",
            route_after_retrieval,
            {"rag": "rag", "web_search": "web_search"}
        )

        graph.add_edge("rag", "summarizer")
        graph.add_edge("web_search", "summarizer")
        graph.add_edge("summarizer", "memory")
        graph.add_edge("memory", END)

        rag_app = graph.compile()

        logger.info("AgenticRAG system initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------
def get_memory_summary():
    """Get a summary of memory usage"""
    if not memory:
        return {"error": "Memory not initialized"}

    messages = memory.chat_memory.messages
    return {
        "total_messages": len(messages),
        "human_messages": len([m for m in messages if isinstance(m, HumanMessage)]),
        "ai_messages": len([m for m in messages if isinstance(m, AIMessage)]),
        "memory_window_size": memory.k
    }

def process_uploaded_file(file: UploadFile) -> int:
    """Process uploaded file and add to vector store"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.filename.lower().endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            loader = UnstructuredLoader(tmp_path)

        documents = loader.load()
        if not documents:
            raise ValueError("No documents could be loaded from the file")

        for doc in documents:
            doc.metadata["source"] = file.filename

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        if not docs:
            raise ValueError("No text chunks could be created from the documents")

        vector_store.add_documents(docs)
        logger.info(f"Successfully added {len(docs)} chunks to vector store")

        return len(docs)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up temporary file {tmp_path}: {cleanup_error}")

# -----------------------------------------------------
# Startup Event
# -----------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    logger.info("Starting AgenticRAG FastAPI application...")
    success = initialize_rag_system()
    if not success:
        logger.error("Failed to initialize RAG system. Some features may not work.")

# -----------------------------------------------------
# API Endpoints
# -----------------------------------------------------
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic information"""
    return HealthResponse(
        status="healthy",
        message="AgenticRAG API is running. Visit /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="AgenticRAG API is healthy"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint using the full RAG graph workflow"""
    if not rag_app:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_app.invoke({"query": request.query})
        memory_summary = get_memory_summary()

        return ChatResponse(
            query=request.query,
            answer=result.get("answer", "No answer generated"),
            source=result.get("source", "Unknown"),
            session_id=request.session_id,
            conversation_length=memory_summary.get("total_messages", 0),
            memory_summary=memory_summary
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload documents to the vector store"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        chunks_added = process_uploaded_file(file)
        return UploadResponse(
            message=f"File '{file.filename}' uploaded successfully!",
            filename=file.filename,
            total_chunks_added=chunks_added
        )

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get comprehensive system status"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        collection = vector_store._collection
        document_count = collection.count()

        sample_docs = []
        if document_count > 0:
            results = vector_store.similarity_search("", k=min(5, document_count))
            sample_docs = [doc.metadata.get("source", "Unknown") for doc in results]

        memory_status = get_memory_summary()

        return StatusResponse(
            vector_store_status="active" if vector_store else "inactive",
            document_count=document_count,
            sample_documents=sample_docs,
            memory_status=memory_status
        )

    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.get("/memory")
async def get_memory():
    """Get current memory status and recent messages"""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        messages = memory.chat_memory.messages
        recent_messages = []

        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                recent_messages.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                recent_messages.append({"type": "ai", "content": msg.content})

        return {
            "memory_summary": get_memory_summary(),
            "recent_messages": recent_messages
        }

    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting memory: {str(e)}")

@app.delete("/memory")
async def clear_memory():
    """Clear the conversation memory"""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    try:
        memory.clear()
        logger.info("Memory cleared")
        return {"message": "Memory cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

# -----------------------------------------------------
# Run Server
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
