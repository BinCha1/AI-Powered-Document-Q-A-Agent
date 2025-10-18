# AgenticRAG

A comprehensive Retrieval-Augmented Generation (RAG) system that combines document processing, intelligent retrieval, conversation routing, and web search fallback capabilities. Built with LangGraph, FastAPI, and React.

## Project Overview

AgenticRAG is an intelligent document Q&A system that goes beyond traditional RAG by incorporating:

- **Intelligent Routing**: Automatically chooses between document retrieval and web search
- **Multi-modal Document Support**: Handles PDF, TXT, DOCX, and other formats
- **Web Search Fallback**: Uses SerpAPI when documents don't contain answers
- **Modern Web Interface**: React frontend with FastAPI backend
- **Persistent Storage**: ChromaDB for vector storage

## 🏗️ Architecture & Flow

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  LangGraph Core │
│                 │    │                 │    │                 │
│ • File Upload   │◄──►│ • REST API      │◄──►│ • RAG Pipeline  │
│ • Chat Interface│    │ • Document Proc │    │ • Routing Logic │
│ • Response Display│   │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                        │
                        ▼
                  ┌─────────────────┐
                  │   External APIs │
                  │                 │
                  │ • Groq LLM      │
                  │ • SerpAPI       │
                  │ • ChromaDB      │
                  └─────────────────┘
```

## Project Structure

```

AgenticRAG/
├── app.py # FastAPI backend server
├── requirements.txt # Python dependencies
├── notebook/
│ ├── agenticrag.ipynb # Core RAG system development
│ ├── test.pdf # Sample document
│ └── chroma_db/ # Vector database
├── frontend/
│ ├── src/
│ │ └── App.js # React frontend
│ ├── package.json # Node.js dependencies
│ └── public/
└── chroma_db/ # Main vector database

```

1. **Clone and navigate to project**:

   ```bash
   cd AgenticRAG
   ```

````

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   ```

4. **Run the FastAPI server**:
   ```bash
   python app.py
   ```
   Server will start at `http://127.0.0.1:8000`

### Frontend Setup

1. **Navigate to frontend directory**:

   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:

   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```
   Frontend will be available at `http://localhost:3000`

## Backend Setup

1. **Navigate to project root**:

   ```bash
   cd AgenticRAG
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SERPAPI_API_KEY=your_serpapi_key_here
   ```
4. **Run the FastAPI server**:
   ```bash
   python app.py
   ```
   Server will start at `http://127.0.0.1:8000`

### Web Interface

1. **Upload Documents**: Use the file upload form to add PDF, TXT, or DOCX files
2. **Ask Questions**: Type questions in the chat interface
3. **View Responses**: See answers with source attribution (RAG or Web Search)

## Acknowledgments

- **LangChain**: For the RAG framework
- **LangGraph**: For the workflow orchestration
- **Groq**: For fast LLM inference
- **ChromaDB**: For vector storage
- **FastAPI**: For the REST API framework
- **React**: For the frontend interface
````
