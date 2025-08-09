# %%
# https://python.langchain.com/docs/integrations/vectorstores/
from uuid import uuid4
from dotenv import load_dotenv
import os
load_dotenv()

# from pdf_readers import read_pdf
# data = read_pdf(['2306.03514v3.pdf'], 'pymupdf4llm', 'Camelot', 'None')
# from chunkers import *
# chunks1 = create_document_chunks(data)
                                

# with open("example.txt", "r") as file:
#     text = file.read()
# from chunkers import *
# chunks = create_document_chunks([{'text':text, 'tables':[]}])[:5]

# from embedding_model import *
# embedding_model = get_embedding_model('HuggingFace')#OpenAI, ollama, huggingFace

# %%
# https://python.langchain.com/docs/integrations/vectorstores/faiss/
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

def create_faiss_db(embedding_model, chunks):
    # add embedding model
    embedding_dim = len(embedding_model.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # add dcouments
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    
    vector_store.add_documents(documents=chunks, ids=uuids)

    # save vector store
    vector_store.save_local("faiss_index")
    return vector_store


# %%
# https://python.langchain.com/docs/integrations/vectorstores/chroma/
import chromadb
from langchain_chroma import Chroma

def create_chroma_db(embedding_model, chunks):
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_model,
        # persist_directory="chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

    # add dcouments
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=uuids)

    return vector_store


# %%
# based on Approximate nearest neighbour (ANN) search
# https://python.langchain.com/docs/integrations/vectorstores/weaviate/
# https://docs.weaviate.io/weaviate/client-libraries/python
# create an account on Weaviate Cloud Service console (WCS console) and create a cluster there
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore

def create_weaviate_db(embedding_model, chunks):
    
    # Access environment variables
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')

    weaviate_url = "https://ghzgtzddteodpiezzvszba.c0.asia-southeast1.gcp.weaviate.cloud"

    weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )

    # weaviate_client = weaviate.connect_to_local()
    vector_store = WeaviateVectorStore.from_documents(chunks, 
                                            embedding_model, 
                                            client=weaviate_client,
                                            # tenant="user1" # when creating separate data for multiusers
                                            )
    return vector_store


# %%
# an inverted index maps each unique word or term to the locations (such as document IDs or row numbers) where it appears. This setup allows for very efficient retrieval of information, especially in large datasets.
# https://python.langchain.com/docs/integrations/vectorstores/activeloop_deeplake/
# https://docs.deeplake.ai/latest/
from langchain_deeplake.vectorstores import DeeplakeVectorStore

def create_activeloop_db(embedding_model, chunks):
    vector_store = DeeplakeVectorStore.from_documents(chunks, 
                                                  dataset_path="my_deeplake/", 
                                                  embedding=embedding_model, 
                                                  overwrite=True)

    return vector_store


# %%
# https://python.langchain.com/docs/integrations/vectorstores/pinecone/
# https://www.pinecone.io/learn/hybrid-search-intro/
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone
from pinecone import ServerlessSpec

def create_pinecone_db(embedding_model, chunks, search_type):
    # Access environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')

    pc = Pinecone(api_key=pinecone_api_key)

    embedding_dim = len(embedding_model.embed_query("hello world"))

    if search_type != 'hybrid search':
        index_name = "sample-index"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=embedding_dim, # embedding model dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

        uuids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=uuids)
    
    else:
        index_name = "sample-index-hybrid"
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=embedding_dim, # embedding model dimension
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        index = pc.Index(index_name)

        # or from pinecone_text.sparse import SpladeEncoder if you wish to work with SPLADE
        # use default tf-idf values
        bm25_encoder = BM25Encoder().default()
        corpus = [chunk.page_content for chunk in chunks]
        bm25_encoder.fit(corpus)

        # store the values to a json file
        bm25_encoder.dump("bm25_values.json")
        # load to your BM25Encoder object
        bm25_encoder = BM25Encoder().load("bm25_values.json")

        vector_store = PineconeHybridSearchRetriever(
            embeddings=embedding_model, 
            sparse_encoder=bm25_encoder, 
            index=index,
            alpha=1,
            top_k=3
        )

        # add text
        vector_store.add_texts(corpus)
    
    return vector_store


# %%
# hybrid search- weaviate, pinecone
def get_vector_db(vector_db, embedding_model, chunks, search_type=''):
    if vector_db == 'FAISS':
        vector_store = create_faiss_db(embedding_model, chunks)
    if vector_db == 'ChromaDB':
        vector_store = create_chroma_db(embedding_model, chunks)
    if vector_db == 'Weaviate':
        vector_store = create_weaviate_db(embedding_model, chunks)
    if vector_db == 'ActiveLoop':
        vector_store = create_activeloop_db(embedding_model, chunks)
    if vector_db == 'Pinecone':
        vector_store = create_pinecone_db(embedding_model, chunks, search_type)
    return vector_store

