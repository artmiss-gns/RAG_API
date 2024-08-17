import os
from pathlib import Path
from pprint import pprint

import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    get_response_synthesizer,
    StorageContext,
    load_index_from_storage,

)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import(
    VectorIndexRetriever,
    # SummaryIndexRetriever,
    # TransformRetriever,
)
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.groq import Groq
from llama_parse import LlamaParse

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# API keys
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
LLAMA_CLOUD_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]

def load_documents(doc_path: Path) -> list[Document]:
    """
    Load and parse documents using LlamaParse.
    """
    file_type = doc_path.suffix
    parser = LlamaParse(result_type="text")
    file_extractor = {file_type: parser}
    documents = SimpleDirectoryReader(input_files=[doc_path], file_extractor=file_extractor).load_data()
    return documents

def setup_llm():
    llm = Groq(
        model="llama3-groq-70b-8192-tool-use-preview",
        api_key=GROQ_API_KEY
    )
    Settings.llm = llm 
    return llm

def setup_embedding():
    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    Settings.embed_model = embed_model

def create_index(documents, embed_model, save_index=False):
    index = VectorStoreIndex.from_documents(documents, model=embed_model)
    if save_index:
        index.storage_context.persist(persist_dir=f"data/.index/{documents[0].metadata["file_name"]}") # data/stored/file_name
    return index

def create_query_engine(index, k=3):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,
    )

    response_synthesizer = get_response_synthesizer()

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    return query_engine

def run(doc_path, rebuild_index=True, index_path=None, save_index=False):
    embed_model = setup_embedding() 
    setup_llm() # setting up the global llm

    # loading the index
    if rebuild_index is False: 
        print("Loading Index...")
        if index_path is None:
            raise ValueError("index_path is required if rebuild_index is False")
        storage_context = StorageContext.from_defaults(persist_dir=index_path) # rebuild storage context
        index = load_index_from_storage(storage_context)
    # rebuilding the index
    else : 
        print("Rebuilding Index...")
        documents = load_documents(doc_path)
        index = create_index(documents, embed_model, save_index)

    query_engine = create_query_engine(index, k=5) #! if the found documents is less than 5 ?

    query = input("-> : ")
    response = query_engine.query(query)

    return response

if __name__ == "__main__":
    doc_path = Path("data/Academic-CV-V1 .pdf")
    response = run(
        doc_path,
        # save_index=True,
        rebuild_index=False,
        index_path=f"data/.index/{doc_path.name}", # index name is the name of the uploaded file
    )
    pprint(response.response)