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

    def __call__(self, query, documents, rebuild_index=False, save_index=False):
        self.documents = documents
        # loading the index
        if rebuild_index is False: 
            print("\nLoading Index...\n")
            self.load_index()
        # rebuilding the index
        else : 
            print("\nBuilding Index...\n")
            self.create_index(save_index)

        query_engine = self.create_query_engine(k=5) #! if the founded documents is less than 5 ??
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

    def load_index(self):
        """
        # ! Note :
        Currently the loading is only based on the file name. Later it should be changed to something else.
        """
        file_name = self.documents[0].metadata["file_name"]
        if file_name is None:
            raise ValueError("Index not found.")
        storage_context = StorageContext.from_defaults(persist_dir=f"data/.index/{file_name}") # rebuild storage context
        self.index = load_index_from_storage(storage_context)

    def create_index(self, save_index=False):
        index = VectorStoreIndex.from_documents(self.documents, model=self.embed_model)
        if save_index:
            index.storage_context.persist(persist_dir=f"data/.index/{self.documents[0].metadata["file_name"]}") # data/stored/file_name
        self.index = index

    def setup_llm(self):
        self.llm = Groq(
            model="llama3-groq-70b-8192-tool-use-preview",
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
    doc_path = Path("data/Academic-CV-V1 .pdf")
    document_retriever = DocumentRetriever(doc_path)
    documents = document_retriever.load_documents()

    # Initialize RAG and run the query
    query = "What are the skills?"
    rag = RAG()
    response = rag(
        query,
        documents,
        # rebuild_index=True,
        # save_index=True
    )

    print(response)