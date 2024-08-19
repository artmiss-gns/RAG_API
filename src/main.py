from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import requests
from pathlib import Path
from src.RAG import RAG, DocumentRetriever
from src.models.models import RAGRequest, RAGResponse
import os

def get_relevant_context(query, context_file_path):
    print("Preprocessing Documents...\n")
    print(context_file_path)
    document_retriever = DocumentRetriever(context_file_path)
    documents = document_retriever.load_documents()

    return documents

def generate_response(context, query, rebuild_index=True, save_index=False):
    rag = RAG()
    response = rag(
        query,
        documents=context,
        rebuild_index=rebuild_index,
        save_index=save_index,
        k=5,
    )

    return response.response

app = FastAPI()

@app.get("/")
async def main_root():
    return {"message": "Welcome to the RAG API"}

@app.post("/", response_model=RAGResponse)
async def rag_endpoint(
    context: UploadFile = File(...),
    query: str = Form(...),
    rebuild_index: bool = Form(True),
    save_index: bool = Form(False),
):
    context_file_path = Path(f"data/{context.filename}")
    if not os.path.exists("data"): # check if data folder is note created 
        os.makedirs("data")
    try:
        with open(context_file_path, "wb") as file:
            file.write(context.file.read())

        context = get_relevant_context(query, context_file_path)
        response = generate_response(context, query, rebuild_index=rebuild_index, save_index=save_index)
        return RAGResponse(answer=response)
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response format from LLM API")
    finally:
        context_file_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8090)
