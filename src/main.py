from fastapi import FastAPI, HTTPException
import requests
from pathlib import Path
from src.RAG import RAG, DocumentRetriever
from src.models.models import RAGRequest, RAGResponse

def get_relevant_context(query):
    print("Preprocessing Documents...\n")
    doc_path = Path("data/Academic-CV-V1 .pdf")
    document_retriever = DocumentRetriever(doc_path)
    documents = document_retriever.load_documents()

    return documents

def generate_response(context, query, rebuild_index=True, save_index=True):
    rag = RAG()
    response = rag(
        query,
        documents=context,
        rebuild_index=rebuild_index,
        save_index=save_index
    )

    return response

app = FastAPI()

@app.get("/")
async def main_root(request):
    return {"message": "Welcome to the RAG API"}

@app.post("/", response_model=RAGResponse)
async def rag_endpoint(request: RAGRequest):
    query = request.query

    try:
        context = get_relevant_context(query)
        response = generate_response(context, query, rebuild_index=True, save_index=True)

        return RAGResponse(answer=response)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response format from LLM API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8090)