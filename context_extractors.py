# %%
def get_context_from_faiss(vector_store, query):
    # load faiss vector store from local
    # vector_store = FAISS.load_local(
    #     "faiss_index", embedding_model, 
    #     allow_dangerous_deserialization=True
    # )

    # Similarity search with filtering on metadata
    # results = vector_store.similarity_search_with_score(
    #     "Abstract of State-of-the-art object detection ?",
    #     k=2,
    #     # filter={"source": "tweet"},
    # )

    # Similarity search with score
    # for res, score in results:
    #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    # Query by turning into retriever
    retriever = vector_store.as_retriever(search_type="mmr", 
                                          search_kwargs={'k': 3, 'fetch_k': 50}
                                          )
    results = retriever.invoke(query, 
        #  filter={"source": "news"}
        )
    
    return results


# %%
def get_context_from_chroma(vector_store, query):
   
    # Similarity search with filtering on metadata
    # results = vector_store.similarity_search_with_score(
    #     "Abstract of State-of-the-art object detection ?",
    #     k=2,
    #     # filter={"source": "tweet"},
    # )

    # Similarity search with score
    # for res, score in results:
    #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")


    # Query by turning into retriever
    retriever = vector_store.as_retriever(search_type="mmr", 
                                          search_kwargs={'k': 3, 'fetch_k': 50}
                                          )
    results = retriever.invoke(query, 
        #  filter={"source": "news"}
        )
    
    return results


# %%
def get_context_from_weaviate(vector_store, query, search_type):
    # query = "Abstract of paper ?"
    # docs = vector_store.similarity_search_with_score(query, k=3,
    #                                         # tenant="user1"
    #                                         )

    # Print the first 100 characters of each result
    # for doc in docs:
    #     print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")


    if search_type!='hybrid search':
        retriever = vector_store.as_retriever(search_type="mmr",
                                                search_kwargs={'k':3, 
                                                               'fetch_k':50, 
                                                               'alpha':0})
        results = retriever.invoke(query)
    else:
    # HYBRID
    # https://docs.weaviate.io/weaviate/api/graphql/search-operators#hybrid
    # alpha can be any number from 0 to 1, defaulting to 0.75.
    # alpha = 0 forces using a pure keyword search method (BM25)
    # alpha = 1 forces using a pure vector search method
    # alpha = 0.5 weighs the BM25 and vector methods evenly
    # fusionType can be rankedFusion or relativeScoreFusion
    # rankedFusion (default) adds inverted ranks of the BM25 and vector search methods
    # relativeScoreFusion adds normalized scores of the BM25 and vector search methods

        results = vector_store.similarity_search(query, 
                                                k=3,
                                                alpha=0.5)

    return results


# %%
def get_context_from_activeloop(vector_store, query):
    # query = "What is the paper abstract ?"

    # load dataset
    # db = DeeplakeVectorStore(
    #     dataset_path="./my_deeplake/", 
    #     embedding_function=embedding_model, r
    #     ead_only=True
    # )

    results = vector_store.similarity_search(
        query,
        k = 3,
        alpha=0.5,
        # filter={"metadata": {"year": 2013}},
    )
    return results


# %%
def get_context_from_pinecone(vector_store, query, search_type):
    # query = "abstract of paper ?"

    if search_type != 'hybrid search':
        # results = vector_store.similarity_search_with_score(
        #     query, k=1, 
        #     # filter={"source": "news"}
        # )
        # for res, score in results:
        #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")    

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, 
                           "score_threshold": 0.4},
        )

        results = retriever.invoke(query, 
                        #  filter={"source": "news"}
                         )
    else:
        results = vector_store.invoke(query, 
                    #  filter={"source": "news"}
                    )

    return results


# %%
def get_context(vector_db, vector_store, query, search_type=''):
    if vector_db=='FAISS':
        context = get_context_from_faiss(vector_store, query)
    if vector_db=='ChromaDB':
        context = get_context_from_chroma(vector_store, query)
    if vector_db=='Weaviate':
        context = get_context_from_weaviate(vector_store, query, search_type)
    if vector_db=='Activeloop':
        context = get_context_from_activeloop(vector_store, query)
    if vector_db=='Pinecone':
        context = get_context_from_pinecone(vector_store, query, search_type)
    return context

