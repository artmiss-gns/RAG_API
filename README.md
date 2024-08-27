# RAG API with Dynamic Document Processing

## 🚨 Important Notice 🚨

🔔 **New Feature Added!** This project now includes a fully functional **User Interface (UI)** that complements the API. The UI allows you to interact with the API directly without needing to write any code.

### 🚀 **Explore the UI Now: [UI Link](https://rag-ui.streamlit.app/)** 🚀

🔗 **Quick Access**: [Installation](#installation) | [Usage](#usage) | [API Endpoints](#api-endpoints)

Please make sure to review the updated documentation to take full advantage of the UI.



## Project Description

This project implements a Retrieval-Augmented Generation (RAG) API using LlamaIndex and FastAPI. It's designed for efficient text-based question answering, integrating LlamaIndex for indexing and LlamaParse for flexible text preprocessing. The service is containerized using Docker for scalability, and a web UI is being developed with Streamlit for seamless user interaction.

## Features

- RAG API using FastAPI for efficient question answering
- Document processing with LlamaParse
- Indexing and retrieval using LlamaIndex
- Containerized service with Docker
- Web UI (in development) using Streamlit

## Technologies Used

- Python
- FastAPI
- LlamaIndex
- LlamaParse
- Docker
- Streamlit (for UI)
- Groq (LLM)
- Cohere (Embeddings)

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
```

## API Endpoints

- GET `/`: Welcome message
- POST `/`: Main RAG endpoint
- Parameters:
    - `context`: File upload (document for context)
    - `query`: String (question to ask)
    - `rebuild_index`: Boolean (optional)
    - `save_index`: Boolean (optional) 

## Deployment

The API is deployed at: https://growing-bessy-hossein-golmohammadi-03788de4.koyeb.app/

## License
MIT License