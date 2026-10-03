# Adaptive RAG - Agentic AI Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange.svg)](https://python.langchain.com/docs/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-purple.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)

---

## 📋 Overview

**Adaptive RAG** is an intelligent, end-to-end Retrieval-Augmented Generation (RAG) system powered by an adaptive LangGraph workflow. It combines dynamic query classification, document retrieval, relevance grading, query rewriting, general LLM responses, and web search to provide context-aware answers.

The system analyzes each user query and dynamically selects one of three processing paths:

* **Index** – use information from uploaded documents
* **General** – answer using the LLM's general knowledge
* **Search** – perform an external web search when current or external information is required

The application consists of a **FastAPI backend** and a **Streamlit frontend**. Uploaded PDF and TXT documents are processed, split into chunks, embedded using HuggingFace embeddings, and indexed in a FAISS vector store for retrieval.

---

## 🎯 Key Features

### 🧠 Intelligent Query Routing

* **Adaptive Classification**: Automatically classifies incoming queries before selecting the processing path.
* **Three Query Types**:

  * **Index**: Queries answerable using the uploaded document collection
  * **General**: Queries answerable using general LLM knowledge or casual conversation
  * **Search**: Queries requiring external or real-time information
* **Context-Aware Routing**: The classifier first checks retrieved indexed context when deciding whether the query belongs to the document-based path.

### 📚 Advanced RAG Pipeline

* **Document Processing**: Supports PDF and TXT files.
* **Text Chunking**: Uses `RecursiveCharacterTextSplitter`.
* **Chunk Configuration**:

  * Chunk size: `1000`
  * Chunk overlap: `150`
* **Embeddings**: Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
* **Vector Search**: Uses FAISS for similarity-based document retrieval.
* **Relevance Grading**: Retrieved context is evaluated using an LLM-based grading step.
* **Query Rewriting**: If retrieved context is not relevant, the query is rewritten and retrieval is performed again.
* **Context-Based Generation**: Relevant retrieved context is passed to the LLM to generate the final response.

### 🤖 Agentic AI Architecture

* **LangGraph Workflow**: The RAG workflow is orchestrated using LangGraph.
* **Adaptive Routing**: Different graph paths are selected according to query classification.
* **Direct Retrieval**: The current retrieval node directly invokes the FAISS retriever.
* **Web Search Integration**: Tavily is used for queries requiring external information.
* **LLM Generation**: The configured Groq LLM is used for classification, grading, rewriting, generation, and general responses.

### 💾 State Management

* **MongoDB Backend**: Chat history is stored using the MongoDB-backed chat history implementation.
* **Session Tracking**: Each Streamlit conversation is associated with a generated UUID session ID.
* **Conversation Context**: Previous messages for the session are retrieved before processing a new query.

### 🎨 User Interface

* **Streamlit Web App**: Interactive web interface for chatting with the RAG system.
* **Document Upload**: Upload PDF and TXT documents directly through the application.
* **Session-Based Conversations**: Conversations are tracked using a generated session ID.
* **Chat Interface**: Users can submit questions and receive responses from the FastAPI backend.

### ⚡ API-First Architecture

* **FastAPI Backend**: Provides REST endpoints for document upload and RAG queries.
* **RESTful Endpoints**:

  * `POST /rag/query`
  * `POST /rag/documents/upload`
* **Streamlit API Client**: The frontend communicates with the FastAPI backend over HTTP.

---

## 🏗️ Architecture

### System Components

```text
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Streamlit Application                  │  │
│  │  • Chat Interface                                         │  │
│  │  • Document Upload (PDF, TXT)                             │  │
│  │  • Session ID Management                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    REST API Endpoints                     │  │
│  │  • POST /rag/query                                        │  │
│  │  • POST /rag/documents/upload                             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                           │
│                                                                 │
│                 ┌─────────────────────┐                         │
│                 │   Query Analysis    │                         │
│                 │    + Classification│                         │
│                 └──────────┬──────────┘                         │
│                            ↓                                    │
│                   ┌────────┴────────┐                           │
│                   │   Route Query   │                           │
│                   └────────┬────────┘                           │
│                            │                                    │
│          ┌─────────────────┼─────────────────┐                  │
│          ↓                 ↓                 ↓                  │
│    ┌───────────┐     ┌────────────┐    ┌────────────┐          │
│    │ Retriever │     │ General LLM│    │ Web Search │          │
│    │  (FAISS)  │     │            │    │  (Tavily)  │          │
│    └─────┬─────┘     └──────┬─────┘    └──────┬─────┘          │
│          ↓                   │                 ↓                │
│    ┌───────────┐             │           ┌───────────┐         │
│    │   Grade   │             │           │  Generate │         │
│    └─────┬─────┘             │           └─────┬─────┘         │
│          ↓                   │                 │                │
│   ┌──────┴──────┐            │                 │                │
│   │             │            │                 │                │
│  yes            no           │                 │                │
│   ↓              ↓           │                 │                │
│ Generate       Rewrite ──────┘                 │                │
│   │              │                             │                │
│   │              └──→ Retriever → Grade ───────┘                │
│   │                                                            │
│   └───────────────────────┬────────────────────────────────────┘
│                           ↓
│                    Final Response
└─────────────────────────────────────────────────────────────────┘
```

### Graph Nodes

1. **query_analysis**: Retrieves indexed context and classifies the incoming query.
2. **retriever**: Retrieves relevant document chunks from the FAISS vector store.
3. **grade**: Evaluates whether the retrieved context is relevant to the query.
4. **rewrite**: Rewrites the query when the retrieved context is not relevant.
5. **generate**: Generates the final answer from the retrieved or searched context.
6. **web_search**: Performs external web search using Tavily.
7. **general_llm**: Answers queries using the configured LLM without document retrieval.

### Document Processing Flow

```text
Uploaded PDF/TXT
       ↓
Temporary File
       ↓
Document Loader
       ↓
Text Extraction
       ↓
RecursiveCharacterTextSplitter
       ↓
Document Chunks
       ↓
HuggingFace Embeddings
       ↓
FAISS Vector Store
       ↓
Retriever
```

The FAISS vector store is currently maintained in application memory. Uploading a document initializes the vector store from the generated document chunks.

---

## 📦 Project Structure

```text
Adaptive rag/
├── src/                              # Main source code
│   ├── main.py                       # FastAPI application entry point
│   ├── api/                          # API routes and endpoints
│   │   └── routes.py                 # RAG query and document upload endpoints
│   ├── config/                       # Configuration and prompts
│   │   ├── settings.py               # Application configuration
│   │   └── prompts.yaml              # LLM prompts
│   ├── core/                         # Core configuration
│   │   └── config.py                 # Environment-based configuration
│   ├── db/                           # Database layer
│   │   └── mongo_client.py           # MongoDB client
│   ├── llms/                         # Language model integrations
│   │   └── openai.py                 # Configured LLM initialization
│   ├── memory/                       # Chat memory management
│   │   ├── chat_history_mongo.py     # MongoDB-backed chat history
│   │   └── chathistory_in_memory.py  # In-memory chat history
│   ├── models/                       # Data models and schemas
│   │   ├── state.py                  # LangGraph state definition
│   │   ├── query_request.py          # Query request schema
│   │   └── verification_result.py    # Verification result model
│   ├── rag/                           # RAG pipeline implementation
│   │   ├── graph_builder.py           # LangGraph workflow construction
│   │   ├── retriever_setup.py         # FAISS and embedding setup
│   │   ├── document_upload.py         # Document processing and indexing
│   │   └── reAct_agent.py             # ReAct agent implementation
│   └── tools/                         # Graph and utility tools
│       ├── common_tools.py             # Shared utility functions
│       └── graph_tools.py              # Graph routing and decision tools
│
├── streamlit_app/                    # Streamlit web application
│   ├── home.py                       # Streamlit home page and session setup
│   ├── pages/                        # Multi-page application
│   │   └── chat.py                   # Chat interface and document upload
│   └── utils/                        # Streamlit utilities
│       └── api_client.py             # FastAPI backend client
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── description.txt                   # Generated document description
└── adaptive_RAG.png                  # Architecture/project image
```

---

## 🔌 API Endpoints

### Base URL

```text
http://localhost:8000
```

### 1. Query Endpoint

**Process a RAG query and get an intelligent response**

```http
POST /rag/query
Content-Type: application/json
```

Request:

```json
{
  "query": "What is the main topic of the document?",
  "session_id": "user_session_123"
}
```

**Response:**

```json
{
  "result": {
    "type": "ai",
    "content": "Based on the document, the main topic is..."
  }
}
```

**Parameters:**

* `query` (string, required): User's question or query.
* `session_id` (string, required): Unique session identifier used for conversation history.

**Status Codes:**

* `200`: Success
* `400`: Invalid request format
* `500`: Server error

---

### 2. Document Upload Endpoint

**Upload a document for RAG indexing**

```http
POST /rag/documents/upload
X-Description: Brief description of the document
```

Form Data:

```text
file: <PDF or TXT file>
```

**Response:**

```json
{
  "status": true
}
```

**Headers:**

* `X-Description` (string, required): Description of the uploaded document.

**Parameters:**

* `file` (file, required): PDF or TXT document.

**Supported Formats:**

* PDF (`.pdf`)
* Plain Text (`.txt`)

**Status Codes:**

* `200`: Successfully uploaded and indexed
* `400`: Invalid file type or missing description
* `500`: Document processing or indexing error

---

## 📖 Usage Guide

### 1. Prerequisites

```text
# System Requirements
- Python 3.9 or higher
- MongoDB
- Groq API key
- Tavily API key for web-search queries
```

FAISS runs locally as part of the application and does not require a separate vector database service.

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/satvik7172-alt/RAG.git
cd RAG

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search Configuration
TAVILY_API_KEY=your_tavily_api_key_here
```

Do not commit the `.env` file to Git. It should remain excluded through `.gitignore`.

### 4. Running the Application

**Start FastAPI Backend:**

```bash
# Terminal 1
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Start Streamlit Frontend:**

```bash
# Terminal 2
streamlit run streamlit_app/home.py
```

**Access the Application:**

* Web Interface: `http://localhost:8501`
* API Documentation: `http://localhost:8000/docs`
* ReDoc Documentation: `http://localhost:8000/redoc`

### 5. Example Usage

**Using the Web Interface:**

1. Start the FastAPI backend.
2. Start the Streamlit application.
3. Open `http://localhost:8501`.
4. Upload a PDF or TXT document.
5. Enter a description for the document.
6. Ask questions about the uploaded document.
7. The system classifies the query and routes it through the appropriate LangGraph path.

**Using cURL:**

```bash
# Upload a document
curl -X POST http://localhost:8000/rag/documents/upload \
  -H "X-Description: Sample document about Python" \
  -F "file=@document.pdf"

# Query the RAG system
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about Python",
    "session_id": "user_123"
  }'
```

**Using Python:**

```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/rag/query",
    json={
        "query": "What is Python?",
        "session_id": "user_123"
    }
)

print(response.json())
```

---

## 🔧 Configuration

### Key Configuration Files

#### `src/core/config.py`

The application loads configuration values from environment variables:

```python
GROQ_API_KEY
TAVILY_API_KEY
```

The configuration module loads these values from the `.env` file and exposes them to the application.

#### `src/rag/retriever_setup.py`

The current document retrieval configuration uses HuggingFace embeddings and FAISS:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Documents are converted into a FAISS vector store after upload.

#### `src/config/prompts.yaml`

Contains prompts for:

* **system_prompt**: Instructions for the ReAct agent implementation.
* **classify_prompt**: Query classification logic.
* **grading_prompt**: Retrieved-context relevance evaluation.
* **rewrite_prompt**: Query rewriting.
* **generate_prompt**: Final response generation.
* **verify_prompt**: Answer faithfulness verification.

### Query Routing Logic

The system routes queries based on classification:

```text
Query Classification
├── "index" → FAISS retriever
│                ↓
│              Grade
│             ↙     ↘
│          "yes"     "no"
│            ↓         ↓
│         Generate   Rewrite
│                      ↓
│                  Retriever
│
├── "general" → General LLM
│
└── "search" → Tavily Web Search
                   ↓
                Generate
```

The classifier follows these rules:

* Choose **index** when relevant information exists in the uploaded document context.
* Choose **general** when the query can be answered using general knowledge or casual conversation.
* Choose **search** when external, real-time, or missing information is required.

---

## 🧪 Testing the API

### Using FastAPI Interactive Documentation

1. Navigate to `http://localhost:8000/docs`.
2. Expand the endpoint sections.
3. Click **Try it out**.
4. Enter the required request data.
5. Click **Execute**.
6. Inspect the API response.

### Example Test Cases

**Test 1: Simple Query**

```json
{
  "query": "Hello, how are you?",
  "session_id": "test_user_1"
}
```

This should normally be classified as a general query.

**Test 2: Document-Based Query**

```json
{
  "query": "What topics are covered in the uploaded document?",
  "session_id": "test_user_1"
}
```

This should use the indexed document path when relevant document context is available.

**Test 3: General Knowledge Query**

```json
{
  "query": "What is machine learning?",
  "session_id": "test_user_1"
}
```

This can be handled by the general LLM path when the uploaded document does not provide relevant context.

---

## 🔐 Security Considerations

* Store API keys in `.env` and never commit them.
* Use environment variables for sensitive configuration.
* Keep `.env` excluded through `.gitignore`.
* Validate uploaded file types.
* Validate user input through FastAPI/Pydantic request models.
* Use HTTPS when deploying the application publicly.
* Secure MongoDB with appropriate credentials and access controls.
* Implement authentication and authorization before exposing the application in a production environment.
* Add rate limiting for public deployments.

---

## 🚀 Deployment

### Local Development

```bash
# Run FastAPI development server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Run the Streamlit frontend separately:

```bash
streamlit run streamlit_app/home.py
```

### Production Deployment

A production deployment can run FastAPI with multiple workers:

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The Streamlit frontend should be deployed separately or hosted alongside the backend according to the deployment environment.

### Docker Support (Optional)

A `Dockerfile` and `docker-compose.yml` can be added for containerized deployment.

---

## 📊 Performance Optimization

* **Document Chunking**: Uses configurable chunk size of 1000 characters with 150-character overlap.
* **Vector Search**: Uses FAISS for local similarity search.
* **In-Memory Vector Store**: Avoids requiring an external vector database for the current implementation.
* **Session-Based History**: Retrieves conversation history from MongoDB for each session.
* **Query Routing**: Avoids document retrieval generation paths when the query can be answered generally.
* **Query Rewriting**: Improves retrieval when the first retrieved context is not relevant.
* **Temporary Files**: Uploaded files are processed through temporary files and deleted after processing.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/YourFeature
```

3. Make changes following the project's coding standards.
4. Test the application locally.
5. Commit with a descriptive message:

```bash
git commit -m "feat: Add YourFeature"
```

6. Push your branch:

```bash
git push origin feature/YourFeature
```

7. Open a Pull Request.

### Code Quality

* Follow PEP 8 standards.
* Add docstrings to functions where appropriate.
* Write tests for new functionality.
* Update documentation when architecture changes.
* Keep API keys and other secrets out of source control.

---

## 📚 Technology Stack

| Component                  | Technology                     | Version                          |
| -------------------------- | ------------------------------ | -------------------------------- |
| **LLM Framework**          | LangChain                      | Installed via `requirements.txt` |
| **Workflow Orchestration** | LangGraph                      | Installed via `requirements.txt` |
| **Web Framework**          | FastAPI                        | Installed via `requirements.txt` |
| **ASGI Server**            | Uvicorn                        | Installed via `requirements.txt` |
| **UI Framework**           | Streamlit                      | Installed via `requirements.txt` |
| **Vector Search**          | FAISS                          | Installed via `requirements.txt` |
| **Embeddings**             | HuggingFace `all-MiniLM-L6-v2` | Installed via `requirements.txt` |
| **Chat Database**          | MongoDB                        | Application backend              |
| **Document Processing**    | LangChain Community            | Installed via `requirements.txt` |
| **LLM Provider**           | Groq                           | API                              |
| **Web Search**             | Tavily                         | API                              |
| **Data Validation**        | Pydantic                       | Installed via `requirements.txt` |

---

## 📝 Documentation References

The primary documentation is maintained in this README and in the source-code comments.

Additional documentation files should be added here when they are introduced and committed to the repository.

---

## ❓ FAQ

**Q: What document formats are supported?**
A: The current document upload implementation supports PDF and TXT files.

**Q: How are uploaded documents indexed?**
A: Documents are loaded, split into chunks, embedded using HuggingFace `sentence-transformers/all-MiniLM-L6-v2`, and stored in an in-memory FAISS vector store.

**Q: Do I need Qdrant?**
A: No. The current implementation uses FAISS for vector retrieval and does not require a separate Qdrant server.

**Q: How is conversation history stored?**
A: The query endpoint retrieves and updates session-based chat history through the MongoDB-backed `ChatHistory` implementation.

**Q: How is a session created?**
A: The Streamlit application generates a UUID and stores it in `st.session_state`. This session ID is sent to the FastAPI backend with each query.

**Q: Can I run the application without uploading a document?**
A: Document-based queries require an indexed document because the current FAISS retriever is initialized during document upload. General and search routes do not require a document when those routes are selected.

**Q: Can I use different LLM providers?**
A: The LLM configuration is currently implemented in `src/llms/openai.py` and uses the configured Groq integration. The provider/model configuration can be changed there if the corresponding LangChain integration is installed.

**Q: Does the current retrieval node use a ReAct agent?**
A: No. The current LangGraph retrieval node directly invokes the FAISS retriever. The ReAct implementation remains available in `src/rag/reAct_agent.py`, but it is not used by the current retrieval node.

**Q: Where are uploaded files stored?**
A: Uploaded files are written to temporary files for processing. The temporary file is deleted after processing. The resulting document chunks are stored in the in-memory FAISS vector store.

---

## 💬 Support & Contact

For issues, questions, or suggestions:

* Open an Issue in the project's GitHub repository.
* Review the source code and README documentation.
* Check the FastAPI documentation at `http://localhost:8000/docs`.
* Review application logs when debugging backend or document-processing errors.

---

## 🙏 Acknowledgments

* Built with LangChain and LangGraph.
* Vector search powered by FAISS.
* Embeddings powered by HuggingFace Sentence Transformers.
* LLM capabilities powered by Groq.
* Web search powered by Tavily.
* API backend powered by FastAPI.
* UI powered by Streamlit.
* Chat history supported by MongoDB.
* Thanks to the open-source community.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 👤 Author

**Satvik Bhardwaj**

* GitHub: [@satvik7172-alt](https://github.com/satvik7172-alt)
* Project: [RAG](https://github.com/satvik7172-alt/RAG)

---

## 📈 Project Status

* ✅ FastAPI backend implemented
* ✅ Streamlit frontend implemented
* ✅ PDF and TXT document upload
* ✅ Document chunking and processing
* ✅ HuggingFace embeddings
* ✅ FAISS vector retrieval
* ✅ Query classification
* ✅ Index/general/search routing
* ✅ Retrieved-context relevance grading
* ✅ Query rewriting
* ✅ General LLM path
* ✅ Tavily web-search path
* ✅ MongoDB-backed session chat history
* ✅ Local UUID-based session management
* 🔄 Further testing and refinement in progress
* 🔄 Production hardening and deployment configuration

---

## 🗺️ Roadmap

* [ ] Persistent FAISS index storage
* [ ] Support for multiple independently managed document collections
* [ ] Enhanced context management
* [ ] Improved retrieval evaluation
* [ ] Multi-language support
* [ ] Performance benchmarks
* [ ] Extended LLM provider support
* [ ] Authentication and authorization
* [ ] Real-time collaboration
* [ ] Analytics dashboard
* [ ] Cost optimization
* [ ] Production deployment configuration

---

**Last Updated**: September 30, 2026
**Status**: 🚧 Active Development
**Documentation**: ✅ Updated
