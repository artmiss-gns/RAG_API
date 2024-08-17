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

def setup_llm_and_embedding():
    """
    Set up LLM and embedding models.
    """
    llm = Groq(
        model="llama3-groq-70b-8192-tool-use-preview",
        api_key=GROQ_API_KEY
    )

    embed_model = CohereEmbedding(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_query",
    )

    Settings.llm = llm 
    Settings.embed_model = embed_model
    # Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    # Settings.num_output = 512
    # Settings.context_window = 3900

    return embed_model

def create_index_and_query_engine(documents, embed_model, k=3):
    """
    Create index and query engine.
    """
    index = VectorStoreIndex.from_documents(documents, model=embed_model)

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

def run():
    # Load and process data
    doc_path = Path("data/Academic-CV-V1 .pdf")
    documents = load_documents(doc_path)

    embed_model = setup_llm_and_embedding() # Setup LLM and embedding
    query_engine = create_index_and_query_engine(documents, embed_model) # Create index and query engine

    # Get user query and generate response
    query = input("-> :")
    response = query_engine.query(query)

    return response

if __name__ == "__main__":
    response = run()
    pprint(response.response)