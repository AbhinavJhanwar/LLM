from dataclasses import dataclass
from typing import Literal, Any
import streamlit as st
from datetime import datetime
import pandas as pd
from pdf_readers import *
from chunkers import *
from vector_dbs import *
from context_extractors import *
from embedding_model import *
from llm_query_models import get_question_answer

@dataclass
class Chat:
	message: str
	origin: Literal["ai", "human"]
	message_id: str = '0'
	response_time: float = 0.0
	pending: bool = False

st.set_page_config(page_title="Chat Bot", page_icon="🤖", layout="wide")

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
	
# def stage_container():
#     parent = st.empty() # parent empty exists in all stages
#     parent.empty() # transient empty exists just long enough to clear the screen
#     time.sleep(.01)
#     return parent.container() # actual container replaces transient empty

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

	# indexing = st.selectbox(
    #             "Indexing Method",
    #             ['Detailed', 'Question Based'],
    #         )

	if st.button("Read Document"): 
		if pdf_docs != None:
			# write pdf just read
			with open(pdf_docs.name, "wb") as f:
				f.write(pdf_docs.read())

			with st.spinner("Reading Document..."):
				st.session_state.data = read_pdf([pdf_docs], pdf_reader, table_reader, image_reader)

	chunker = st.selectbox(
                "Chunking Method",
                ['Characters', 'Recursive Characters', 'Token', 'Recursive Token', 
				 'Markdown', 'HTML', 'Semantic', 'Code', 'Json'],
                help="Choose a chunking  methodology from the dropdown. Code & Json is not supported at the moment"
            )
	
	if st.button("Generate Chunks"): 
		with st.spinner("Generating Chunks..."):
			chunks = create_document_chunks(st.session_state.data, chunker)
			st.session_state.chunks = chunks



	######################## create vector database
	cols = st.columns([1,1,1])		
	embedding_method = cols[0].selectbox(
                "Embedding Framework",
                ['HuggingFace', 'OpenAI', 'Ollama'],
            )

	st.session_state.search_type = cols[1].selectbox(
											"Context Search",
											['cosine similarity', 'hybrid search'],
										)

	if st.session_state.search_type == 'hybrid search':
		vector_databases = ['Pinecone', 'Weaviate']
	else:
		vector_databases = ['FAISS', 'ChromaDB', 'Pinecone', 'Weaviate', 'Activeloop']
	
	st.session_state.vector_database = cols[2].selectbox(
														"Vector Database",
														vector_databases,
													)

	if st.button("Generate Database"): 
		with st.spinner('Generating Vector Store...'):
			st.session_state.embedding_model = get_embedding_model(embedding_method)
			st.session_state.vector_store = get_vector_db(st.session_state.vector_database, 
												 st.session_state.embedding_model, 
												 st.session_state.chunks, 
												 st.session_state.search_type)


	###################### other parameters
	cols = st.columns([1,1])
	st.session_state.framework = cols[0].selectbox(
		"LLM Framewok",
		['Groq', 'OpenAI', 'Ollama', 'HuggingFaceEndpoint'],
		help="For Ollama, download the model and run the server locally"
	)
	if st.session_state.framework == 'Ollama':
		models = ['Llama4', 'Llama3.1', 'Llama3.2', 'Llama3',
                'Gemma3', 'Gemma2', 'Deepseek', 'Qwen', 
                'Mistral', 'Vicuna', 'Phi-4']
	elif st.session_state.framework == 'OpenAI':
		models = ['gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-4-turbo']
	elif st.session_state.framework == 'HuggingFaceEndpoint':
		models = ['Mistral', 'Llama', 'Deepseeek', 'Qwen']
	else:
		models = ['Llama4', 'Llama3.1', 'Llama3', 'Gemma2', 'Deepseek', 'Qwen', 'Mistral']
		
	st.session_state.llm_algorithm = cols[1].selectbox(
                "Large Language Model",
                models,
            )

# if 'chunks' in st.session_state:
# 	for chunk in st.session_state.chunks[:1]:
# 		st.write(chunk)

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
				with st.spinner('Generating Context...'):
					context = get_context(st.session_state.vector_database, 
											st.session_state.vector_store, 
											prompt, 
											st.session_state.search_type)
					context = '\n\n\n'.join([document.page_content for document in context])

				with st.spinner('Generating Response...'):
					ai_message = get_question_answer(prompt, context, st.session_state.framework, st.session_state.llm_algorithm)
				# for step in agent_executor.stream(({
				# 								"input": (
				# 									st.session_state.history[-1].message
				# 										)
				# }),):
				# 	try:
				# 		if "output" in step:
				# 			ai_message = step["output"]
				# 		else:
				# 			ai_message = step
			# except:
			# 	ai_message = "Something Went Wrong, Please upload a document and submit"
				
		# clear ai chat text
		ai_chat_container.empty()

		st.markdown(context)

		# add the ai chat in history and make session available
		st.session_state.history.append(Chat(ai_message, "ai", len(st.session_state.history), pending=False))
		chat.pending = False
		st.session_state.available = True


