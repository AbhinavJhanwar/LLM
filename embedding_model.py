# %%
# https://python.langchain.com/docs/integrations/vectorstores/

# %%
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_embedding_model(model_name):
    if model_name == "OpenAI":
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    if model_name == 'HuggingFace':
        # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if model_name == 'Ollama':
        # https://ollama.com/library
        embedding_model = OllamaEmbeddings(model="llama3")

    return embedding_model