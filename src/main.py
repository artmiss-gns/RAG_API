from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
import requests
from pathlib import Path
from src.RAG import RAG, DocumentRetriever
from src.models.models import RAGRequest, RAGResponse
import os

def get_relevant_context(context_file_path):
    print("Preprocessing Documents...\n")
    print(context_file_path)
    document_retriever = DocumentRetriever(context_file_path)
    documents = document_retriever.load_documents()

    return documents


def generate_response(documents, query, load_index=False, save_index=False, index_name=None, k=5):
    rag = RAG()
    response = rag(
        query,
        documents=documents,
        load_index=load_index,
        save_index=save_index,
        index_name=index_name,
        k=k,
    )
    return response.response


def validate_inputs(
    load_index: bool = Form(False),
    save_index: bool = Form(False),
    index_name: str = Form(None)
):
    """
    Validate the inputs.
    """
    if (save_index or load_index) and not index_name:
        raise HTTPException(
            status_code=400,
            detail="index_name is required when save_index or load_index is True"
        )
    elif not (save_index or load_index) and index_name:
        raise HTTPException(
            status_code=400,
            detail="index_name is not required when save_index and load_index are both False"
        )
    if save_index and load_index:
        raise HTTPException(
            status_code=400,
            detail="save_index and load_index cannot both be True"
        )
    return index_name


app = FastAPI()

@app.get("/")
async def main_root():
    return {"message": "Welcome to the RAG API"}


@app.post("/", response_model=RAGResponse)
async def rag_endpoint(
    context: UploadFile = File(...),
    query: str = Form(...),
    # rebuild_index: bool = Form(True),
    load_index: bool = Form(False),
    save_index: bool = Form(False),
    index_name: str = Depends(validate_inputs),
    k: int = Form(5),
):
    context_file_path = Path(f"data/{context.filename}")
    if not os.path.exists("data"): # check if data folder is note created 
        os.makedirs("data")
    try:
        with open(context_file_path, "wb") as file:
            file.write(context.file.read())

        document = get_relevant_context(context_file_path)
        response = generate_response(document, query, load_index=load_index, save_index=save_index, index_name=index_name, k=k)
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
