# %%
# https://www.pinecone.io/learn/chunking-strategies/
# https://python.langchain.com/docs/concepts/text_splitters/

from langchain_core.documents import Document

# with open("example.txt", "r") as file:
#     text = file.read()
# text

# character, token, recursive character, recursive token, markdown, html, json, code, semantic
# %%
## Length-based
# 1. Token-based: Splits text based on the number of tokens, which is useful when working with language models.
# 2. Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.

from langchain_text_splitters import CharacterTextSplitter
from transformers import GPT2TokenizerFast
# https://python.langchain.com/docs/how_to/character_text_splitter/
def split_on_character(text, chunk_size=1000, overlap=200):
    """
    This splits based on a given character sequence, 
    which defaults to "\n\n". Chunk length is measured by number of characters.
    
    How the text is split: by single character separator.
    How the chunk size is measured: by number of characters.
    """

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # split as document along with metadata
    # chunks = text_splitter.create_documents([text], metadatas=[{'document': 'frcnn research paper'}])
    # print(chunks[0])

    # directly split as strings
    # chunks = text_splitter.split_text(text)
    # print(chunks[0])
    if type(text)!=list:
        text = [text]
    chunks = text_splitter.create_documents(text)
    
    return chunks

# https://python.langchain.com/docs/how_to/split_by_token/
def split_on_token(text, token_size=500, overlap=50, tokenizer='tiktoken'):
    """
    1. tiktoken- It will probably be more accurate for the OpenAI models.
        To split with a CharacterTextSplitter and then merge chunks with tiktoken, 
        use its .from_tiktoken_encoder() method. Note that splits from this method 
        can be larger than the chunk size measured by the tiktoken tokenizer.
        CharacterTextSplitter, RecursiveCharacterTextSplitter, and TokenTextSplitter 
        can be used with tiktoken directly.
    2. huggingFace
    
    """
    if tokenizer=='tiktoken':
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=token_size, chunk_overlap=overlap
        )

    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=token_size, chunk_overlap=overlap
        )

    # chunks = text_splitter.split_text(text)
    # print(chunks[0])
    if type(text)!=list:
        text = [text]
    chunks = text_splitter.create_documents(text)
    
    return chunks

# %%
## Text-structured based
# used when strict limit on token_size
# The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact using '\n\n'.
# If a unit exceeds the chunk size, it moves to the next level (e.g., sentences, using '\n').
# This process continues down to the word level if necessary until the text is within the chunk size.

from langchain_text_splitters import RecursiveCharacterTextSplitter
# https://python.langchain.com/docs/how_to/recursive_text_splitter/

def split_on_character_recursive(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    if type(text)!=list:
        text = [text]
    chunks = text_splitter.create_documents(text)
    # chunks = text_splitter.split_text(text)
    # print(chunks[0])
    # print(chunks[1])
    return chunks


def split_on_token_recursive(text, token_size=500, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # Set a really small chunk size, just to show.
        encoding_name="cl100k_base",
        chunk_size=token_size,
        chunk_overlap=overlap
    )
    if type(text)!=list:
        text = [text]

    chunks = text_splitter.create_documents(text)
    # chunks = text_splitter.split_text(text)
    # print(chunks[0])
    # print(chunks[1])
    return chunks

# %%
## Document-structured based
# Preserves the logical organization of the document
# Maintains context within each chunk
# Can be more effective for downstream tasks like retrieval or summarization

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter

# JSON: Split by object or array elements- https://python.langchain.com/docs/how_to/recursive_json_splitter/
# Code: Split by functions, classes, or logical blocks- https://python.langchain.com/docs/how_to/code_splitter/

# https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/
def split_on_markdown(text, headers_to_split_on=
                      [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                    ],
                    token_size=500,
                    overlap=50):
    # Markdown: Split based on headers (e.g., #, ##, ###)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, 
                                                strip_headers=False #this makes sure headers are also included in chunks
                                                )
    md_header_splits = markdown_splitter.split_text(text)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # Set a really small chunk size, just to show.
        encoding_name="cl100k_base",
        chunk_size=token_size,
        chunk_overlap=overlap
    )
        
    chunks = text_splitter.split_documents(md_header_splits)
    
    return chunks


# https://python.langchain.com/docs/how_to/split_html/
def split_on_html(text, headers_to_split_on=
                      [
                        ("h1", "Header 1"),
                        ("h2", "Header 2"),
                        ("h3", "Header 3"),
                    ],
                    token_size=200,
                    overlap=10):
    # HTML: Split using tags
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)
    chunks = html_splitter.split_text(text)
    # it can again be merged with token splitter as in markdown to limit tokens or character
    return chunks


# %%
## Semantic meaning based
# Start with the first few sentences and generate an embedding.
# Move to the next group of sentences and generate another embedding (e.g., using a sliding window approach).
# Compare the embeddings to find significant differences, which indicate potential "break points" between semantic sections.

# https://python.langchain.com/docs/how_to/semantic-chunker/
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def split_on_semantic(text, breakpoint_threshold_type="interquartile",
                                breakpoint_threshold_amount=1.5):
    # interquartile- used with breakpoint_threshold_amount- default value is 1.5
    # gradient- used with breakpoint_threshold_amount- expects a number between 0.0 and 100.0, the default value is 95.0
    # standard_deviation- used with breakpoint_threshold_amount- the default value is 3.0
    # percentile- used with breakpoint_threshold_amount- expects a number between 0.0 and 100.0, the default value is 95.0
    text_splitter = SemanticChunker(OpenAIEmbeddings(),
                                    breakpoint_threshold_type=breakpoint_threshold_type,
                                    breakpoint_threshold_amount=breakpoint_threshold_amount
                                    )
    chunks = text_splitter.create_documents([text])
    return chunks


# %%
def create_document_chunks(documents, methodology='Token'):
    
    for document in documents:
        # add text data
        text = document['text']
        if methodology=='Characters':
            chunks = split_on_character(text)
        if methodology=='Recursive Characters':
            chunks = split_on_character_recursive(text)
        if methodology=='Token':
            chunks = split_on_token(text)
        if methodology=='Recursive Token':
            chunks = split_on_token_recursive(text)
        if methodology=='Markdown':
            chunks = split_on_markdown(text)
        if methodology=='HTML':
            chunks = split_on_html(text)
        if methodology=='Semantic':
            chunks = split_on_semantic(text)
        
        # add table data
        for i, table in enumerate(document['tables']):
            table_chunk = Document(
                                    page_content=table,
                                    metadata={"table": f"Table {i}"}
                                    )
            chunks.append(table_chunk)

    return chunks