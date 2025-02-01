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


class DocumentRetriever:
    def __init__(self, doc_path: Path, result_type="text", ):
        self.doc_path = doc_path
        self.result_type = result_type

    def load_documents(self) -> list[Document]:
        """
        Load and parse documents using LlamaParse.
        """
        file_type = self.doc_path.suffix
        parser = LlamaParse(result_type="text")
        file_extractor = {file_type: parser}
        documents = SimpleDirectoryReader(input_files=[self.doc_path], file_extractor=file_extractor).load_data()
        return documents

class RAG:
    def __init__(self) :
        self.setup_embedding()
        self.setup_llm()

    def __call__(self, query, documents, load_index=False, save_index=False, index_name=None, k=5): # TODO: add constraint for index_name when load_index and save_index are both False
        self.documents = documents
        if load_index:
            print("\nLoading Index...\n")
            self.load_index(index_name=index_name)
        else :
            print("\nBuilding Index...\n")
            self.create_index(save_index=save_index, index_name=index_name)

        query_engine = self.create_query_engine(k=k) #! if the founded documents is less than 5 ??
        response = query_engine.query(query)

        return response

    def create_query_engine(self, k=3):
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )

        response_synthesizer = get_response_synthesizer()

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        return query_engine

    def create_index(self, save_index=False, index_name=None):
        index = VectorStoreIndex.from_documents(self.documents, model=self.embed_model)
        if save_index:
            self.save_index(index, index_name)
        self.index = index

    def save_index(self, index, index_name):
        index.storage_context.persist(persist_dir=f"data/saved_index/{index_name}")

    def load_index(self, index_name):
        storage_context = StorageContext.from_defaults(persist_dir=f"data/saved_index/{index_name}") # rebuild storage context
        self.index = load_index_from_storage(storage_context) # load index
    
    def setup_llm(self):
        self.llm = Groq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY
        )
        Settings.llm = self.llm

    def setup_embedding(self):
        self.embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0",
            input_type="search_query",
        )
        Settings.embed_model = self.embed_model



if __name__ == "__main__":
    # Load document
    print("Preprocessing Documents...\n")
    doc_path = Path("data/Academic-CV-V1.pdf")
    document_retriever = DocumentRetriever(doc_path)
    documents = document_retriever.load_documents()

    # Initialize RAG and run the query
    query = "What are the skills?"
    rag = RAG()
    response = rag(
        query,
        documents,
        # load_index=False,
        # save_index=False,
        # k=5
    )

    print(response)
