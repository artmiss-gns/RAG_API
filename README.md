# EduRAG: RAG-Powered Educational Assistant

## 🎓 Transform Your Learning Experience

EduRAG is an innovative educational platform that combines the power of Retrieval-Augmented Generation (RAG) with an intuitive user interface to create a personalized learning experience. Whether you're studying Mathematics, Science, History, or Computer Science, EduRAG helps you understand, practice, and master your subjects.

### 🚀 [Start Learning Now](https://rag-ui.streamlit.app/) 

🔗 **Quick Links**: [Installation](#installation) | [Features](#features) | [Getting Started](#getting-started) | [EduRAG Website](https://rag-ui.streamlit.app/)

## What Makes EduRAG Special?

EduRAG transforms traditional studying by:
- 📚 Processing your study materials intelligently
- 🤔 Answering questions with context from your materials
- ✍️ Creating custom quizzes to test your knowledge
- 📝 Generating concise, focused summaries
- 🎯 Adapting to different learning styles and subjects

## System Architecture

EduRAG consists of two main components:

### 1. Educational AI Engine (API)
- Built with FastAPI and LlamaIndex for robust document processing
- Uses RAG technology to understand and process educational content
- Powered by Groq for fast, accurate responses
- Utilizes Cohere embeddings for precise content understanding
- Handles multiple document formats (PDF, DOCX, TXT) via LlamaParse

### 2. Interactive Learning Interface (UI)
- Clean, intuitive Streamlit interface
- Three specialized learning modes:
  - 📖 Study Assistant: For detailed explanations and concept clarification
  - 📝 Quiz Mode: For testing understanding and knowledge retention
  - 📑 Summarize Content: For creating concise study materials
- Subject-specific optimization
- Real-time interaction with your study materials

## Technologies Used

- **Core Engine**: Python, FastAPI, LlamaIndex
- **Document Processing**: LlamaParse
- **AI/ML**: Groq (LLM), Cohere (Embeddings)
- **Interface**: Streamlit
- **Deployment**: Docker

## Installation

1. Clone the repository:
```bash
git https://github.com/artmiss-gns/RAG_API 
cd RAG_API
```
2. Set up environment variables:
Create a `.env` file in the root directory and add the following:
```
GROQ_API_KEY=your_groq_api_key
COHERE_API_KEY=your_cohere_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
PORT=your_desired_port
```
## Usage

### Running Locally
- Set the Python path:
```bash
export PYTHONPATH="/$(pwd):$PYTHONPATH"
```
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Start the FastAPI server:
    ```bash
    python src/main.py
    ```
    or
    ```bash
    uvicorn src.main:app --reload --port=8003
    ```
note: you can set the port to any port you want for this method<br>
The API will be available at `http://localhost:8003`

### Using Docker

1. Build the Docker image:
```bash
docker-compose up --build
```

After that, you can call the api with the following command:
```bash
http -f POST \
    http://localhost:8003\
    context@YOUR_FILE_TO_UPLOAD.pdf\
    query="YOUR QUERY"
    load_index=false
    save_index=false
    index_name=
    k=5
```


## API Endpoints

- GET `/`: Welcome message
- POST `/`: Main RAG endpoint
- Parameters:
    - `context`: File upload (document for context)
    - `query`: String (question to ask)
    - `load_index`: Boolean (optional)
    - `save_index`: Boolean (optional) 
    - `index_name`: String (optional)
    - `k`: Integer (optional)

## Deployment

The API is deployed at: https://growing-bessy-hossein-golmohammadi-03788de4.koyeb.app/
```bash
http -f POST \
    https://growing-bessy-hossein-golmohammadi-03788de4.koyeb.app/\
    context@YOUR_FILE_TO_UPLOAD\
    query="YOUR QUERY"\
    load_index=false\
    save_index=false\
    k=5\
    index_name=
```

## License
MIT License