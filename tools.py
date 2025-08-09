# %%
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

search_tool = TavilySearchResults(max_results=1)


# %%
from typing import Optional, Type
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# create tool or import one
class ContextInput(BaseModel):
    """ Input for Context Extractor tool """
    query: str = Field(description="query to match context")

class ContextExtractor(BaseTool):
    name: str = "Context_Extractor"
    description: str = "useful for when you need to fetch relevant context"
    args_schema: Type[BaseModel] = ContextInput

    def _run(
        self, 
        query:str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        # load faiss vector store from local
        vector_store = FAISS.load_local(
            "faiss_index", embedding_model, 
            allow_dangerous_deserialization=True
        )

        # Query by turning into retriever
        retriever = vector_store.as_retriever(search_type="mmr", 
                                          search_kwargs={'k': 3, 'fetch_k': 50}
                                          )
        results = retriever.invoke(query)


        context = '\n\n\n'.join([document.page_content for document in results])
        print(context)

        return context

    async def _arun(
        self,
        query:str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync())
