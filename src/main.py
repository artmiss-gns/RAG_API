from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends
import requests
from typing import Optional
from pathlib import Path
from src.RAG import RAG, DocumentRetriever
from src.models.models import RAGRequest, RAGResponse
import os


def generate_response(query, context=None , load_index=False, save_index=False, index_name=None, k=5):
    if context:
        context_file_path = Path(f"data/context_file_{context.filename}")
        # saving the context file
        with open(context_file_path, "wb") as file:
            file.write(context.file.read())
    else:
        context_file_path = None
    
    print("RAG is being called...") 
    rag = RAG()
    response = rag(
        query,
        documents_path=context_file_path,
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
        
    # checking if the index name already exists
    if save_index and os.path.exists(f"data/saved_index/{index_name}"):
        raise HTTPException(
            status_code=400,
            detail="Index name already exists, please choose a different name"
        )
    return index_name


app = FastAPI()

@app.get("/")
async def main_root():
    return {"message": "Welcome to the RAG API"}


@app.post("/", response_model=RAGResponse)
async def rag_endpoint(
    context: Optional[UploadFile] = File(None),
    query: str = Form(...),
    load_index: bool = Form(False),
    save_index: bool = Form(False),
    index_name: str = Depends(validate_inputs),
    k: int = Form(5),
):
    if load_index and context:
        raise HTTPException(status_code=400, detail="File upload is not allowed when load_index is True.")
    elif not load_index and context is None:
        raise HTTPException(status_code=400, detail="File upload is required when load_index is False.")

    try:
        response = generate_response(query, context, load_index=load_index, save_index=save_index, index_name=index_name, k=k)
        return RAGResponse(answer=response)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=500, detail="Unexpected response format from LLM API")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index not found: {index_name}")
    finally:
        # removing the context file
        if context:
            os.remove(f"data/context_file_{context.filename}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8090)
