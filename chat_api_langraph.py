from dataclasses import dataclass
import os
from typing import Literal, Any
import streamlit as st
from datetime import datetime
import pandas as pd
from pdf_readers import *
from chunkers import *
from vector_dbs import *

from embedding_model import *

############################################# langGraph code #################################

from langgraph.prebuilt import create_react_agent
from tools import ContextExtractor

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGSMITH_TRACING"] = "false"

def initiate_agent():
    # define tool
    ContextExtraction = ContextExtractor()

    tools = [ContextExtraction]
    llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )

    # Create a ReAct agent with the LLM and tools
    agent_executor = create_react_agent(llm, 
                                        tools)
    
    return agent_executor
	
def run_agent(query):

    messages=[
                ("system", """
                You are a helpful assistant.
                You are provided with a user query, you have to generate context using context tool and then based 
                on that context only answer the user question.
                Do not add any other information that is not mentioned in the context.
                """
                ),
                ("user", f"question: '{query}' "),
            ]

    state = agent_executor.invoke({"messages": messages})
    # print out the agent's traces
    print("\n📜 Agent trace:")
    for i, msg in enumerate(state["messages"]):
        print(f"{i+1}. [{msg.type.upper()}] {msg.content.strip() if hasattr(msg, 'content') else msg}")
        print('=====')
    return state['messages'][-1].content

################################################## langGraph code end #################################

agent_executor = initiate_agent()

@dataclass
class Chat:
	message: str
	origin: Literal["ai", "human"]
	message_id: str = '0'
	response_time: float = 0.0
	pending: bool = False

st.set_page_config(page_title="LangGraph Chat Bot", page_icon="🤖", layout="wide")

def initialize_session_state():
    st.session_state.run = True
    st.session_state.history = []
    st.session_state.available = True
	
def on_submit_message():
	if st.session_state.available and st.session_state.human_prompt != "" and st.session_state.run:
		st.session_state.available = False
		message = st.session_state.human_prompt
		st.session_state.history.append(Chat(message, "human", len(st.session_state.history), pending=True))
		st.session_state.human_prompt = ""

if "history" not in st.session_state:
	initialize_session_state()
	
# get new user text
if prompt := st.chat_input("Hello there!"):
	st.session_state.human_prompt = prompt
	on_submit_message()

# with stage_container():
header = st.container()
with header:
    cols = st.columns([100, 12])
    
    # if history is null or chat hasn't started yet then show below content else pass
    if st.session_state.history != []:
        cols[1].button(
            "Clear", 
            # icon=":material/arrow_back:",
            on_click=initialize_session_state
            )

# display sidebar
with st.sidebar:

	######################## read document
	pdf_docs = st.file_uploader("Upload your files", type="pdf", accept_multiple_files=False)

	cols = st.columns([1,1,1])
	pdf_reader = cols[0].selectbox(
                "PDF Reader",
                ['pymupdf4llm', 'PyPDF2', 'pymupdf', 'pytesseract'],
                help="Choose a pdf reader from the dropdown."
            )
	table_reader = cols[1].selectbox(
				"Table Reader", 
				['Tabula', 'Camelot']
			)
	image_reader = cols[2].selectbox(
				"Image Reader",
				['None', 'Ollama', 'OpenAI', 'Groq']
			)

	if st.button("Process Document"): 
		if pdf_docs != None:
			# write pdf just read
			with open(pdf_docs.name, "wb") as f:
				f.write(pdf_docs.read())

			with st.spinner("Reading Document..."):
				st.session_state.data = read_pdf([pdf_docs], pdf_reader, table_reader, image_reader)
				
            ########################## create document chunks
			chunker = 'Recursive Characters'
			with st.spinner("Generating Chunks..."):
				chunks = create_document_chunks(st.session_state.data, chunker)
				st.session_state.chunks = chunks
            
            ######################## create vector database
			st.session_state.embedding_method = "OpenAI"
			st.session_state.search_type = 'cosine similarity'
			st.session_state.vector_database = 'FAISS'
			st.session_state.framework = 'OpenAI'
			st.session_state.llm_algorithm = 'gpt-4o'
			
			with st.spinner('Generating Vector Store...'):
				st.session_state.embedding_model = get_embedding_model(st.session_state.embedding_method)
				get_vector_db(st.session_state.vector_database, 
							st.session_state.embedding_model, 
							st.session_state.chunks, 
							st.session_state.search_type)


# display chat
for chat_number, chat in enumerate(st.session_state.history):
	# for all messages except the last message which is still pending from AI
	if not chat.pending: 
		# display ai message
		if chat.origin == 'ai':
			# check if last message then pass
			with st.chat_message("assistant"):
				st.markdown(chat.message)
				
		# display human message
		else:
			with st.chat_message("user"):
				st.markdown(chat.message)
	
	else:
		with st.chat_message("user"):
			st.markdown(chat.message)

		ai_chat_container = st.empty()
		with ai_chat_container.chat_message("assistant"):
			# try:
				with st.spinner('Generating Output...'):
					ai_message = run_agent(prompt)
					print("query-", prompt)
					print("answer-", ai_message)
					
		# clear ai chat text
		ai_chat_container.empty()

		# add the ai chat in history and make session available
		st.session_state.history.append(Chat(ai_message, "ai", len(st.session_state.history), pending=False))
		chat.pending = False
		st.session_state.available = True


