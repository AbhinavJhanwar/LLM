
## Installation
For general installation-
```
pip install -r requirements.txt
```

for selenium and pytesseract follow below steps-
```
reference- https://github.com/password123456/setup-selenium-with-chrome-driver-on-ubuntu_debian
1. download chrome- wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
2. install chrome- apt-get install -y ./google-chrome-stable_current_amd64.deb
3. check version- google-chrome --version # Google Chrome 129.0.6668.100 
4. sudo apt-get install tesseract-ocr
5. pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

for ollama
```
# download ollama
curl https://ollama.ai/install.sh | sh 
ollama serve # start ollama server/api
ollama pull model_name # download llm in a separate command prompt window
```

for .env file
```
TAVILY_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
GROQ_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
OPENAI_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
WEAVIATE_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
PINECONE_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
LANGSMITH_API_KEY = kgFqlhzBZef5BFZzR9HjeQn2V
LANGSMITH_TRACING = true # based on if you want to trace in langsmith or not
LANGSMITH_PROJECT = project_name
HUGGINGFACEHUB_API_TOKEN = kgFqlhzBZef5BFZzR9HjeQn2V
```

## Run Application
There are two methods-
1. This covers the various pdf readers, chunking methods, vector dbs, and context search type and is mainly a RAG pipeline only and not a conversational bot.<br>
Command to run-
```
streamlit run chat_api.py
```
2. This is a langGraph based chat bot with context extractor as a tool.<br>
Command to run-
```
streamlit run chat_api_langraph.py
```

## PDF Readers
* PyPDF2-  extracts text from pdf in plain text format
* pymupdf4llm- extracts text in markdown format, convert tables into markdown text. It can also save images from the document which can be utilized with multi model framework to analyze. for more details read- https://pymupdf.readthedocs.io/en/latest/ (optimized for images)
* pymupdf - extracts text from pdf in plain text format, extracts tables in markdown format and image in byte format (optimized for tables)
* pytessaract - extract text from any html page or pdf after converting them to images

## Table Readers
* tabula
* camelot

## Image Readers
Uses multimodality using Ollama, Groq or OpenAI framework and loads images and generates a brief summary to be included in pdf. It only supports pymupdf4llm as of now as it will replace the image url with image details in the text.

## Chunking Methods
Various chunking methods are supported, update the length of token, character, overlap as per the usecase.
* recursive character- this splits text on- "\n\n" - Double new line, paragraph breaks, "\n" - New lines, " " - Spaces, "" - Characters. It basically starts with first split, checks the chunk size if it is within boundary if not then further divides chunk on next split type and keeps on going until the chunk size is within boundary
* markdown- this splits text on- \n#{1,6} - Split by new lines followed by a header (H1 through H6)
"```\n" - Code blocks, \n\\*\\*\\*+\n - Horizontal Lines, \n---+\n - Horizontal Lines, \n___+\n - Horizontal Lines, \n\n Double new lines, \n - New line, " " - Spaces, "" - Character.
* python code- this splits text on- \nclass - Classes first, \ndef - Functions next, \n\tdef - Indented functions, \n\n - Double New lines, \n - New Lines, " " - Spaces, "" - Characters
* Semantic- this separates the data into sentences and then for couple of sentences calculates openai embeddings and then looks for cosine similarity between them and keeps on combining the sentences until they have a particular level of similarity. Chunk sizes may not be fixed here.

## Vector Database
Multiple databases are implemented but only weaviate and pinecone support hybrid search.<br>
faiss and chromadb save the database locally and are only suitable for small usecases or POC while others use GCP or Azure or other cloud services to store data and hence better suited for production.


## Context Search
* Cosine- uses cosine similarity between given query and chunks
* Hybrid- uses cosine similarity between given query and chunks but also uses keyword level tfidf based matching between query and chunks using bm25.


## TO DO-
1. using evaluations to determine optimal chunk sizes- https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5
