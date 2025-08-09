# %%
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# %%
############################################### Groq ####################################################
from groq import Groq

def get_groq_response(query, context, model):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    models = {
                'Llama4': 'meta-llama/llama-4-maverick-17b-128e-instruct', 
                'Llama3.1': 'llama-3.1-8b-instant', 
                'Llama3': 'llama3-70b-8192',
                'Gemma2': 'gemma2-9b-it', 
                'Deepseek': 'deepseek-r1-distill-llama-70b', 
                'Qwen': 'qwen/qwen3-32b', 
                'OpenAI': 'whisper-large-v3', 
                'Mistral': 'mistral-saba-24b'
              }

    completion = client.chat.completions.create(
        model=models[model],
        messages=[
                {"role": "system",
                "content": """
                You are a helpful assistant.
                You are provided with context and based on that you have to answer user question.
                Do not add any other information that is not mentioned in the context.
                """
                },
                
                {"role": "user", "content": f"""
                 context: "{context}"
                 question: "{query}" """
                 },
                ],
            )

    return completion.choices[0].message.content


# %%
############################################### ChatGPT ###############################################

import os
from openai import OpenAI

def get_opanai_response(query, context, model):
    # Access environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    client = OpenAI()

    msg = client.responses.create(
        model=model,
        input=[
                {"role": "system",
                "content": """
                You are a helpful assistant.
                You are provided with context and based on that you have to answer user question.
                Do not add any other information that is not mentioned in the context.
                """
                },
                
                {"role": "user", "content": f"""
                 context: "{context}"
                 question: "{query}" """
                 },
            ])
    return msg.output_text


############################################### Ollama ####################################################
# https://ollama.com/download
# https://ollama.com/library

# create server >> ollama serve
# in next commmand prompt window run-
# download model >> ollama pull model_name
# list all downloaded models >> ollama list
# check model properties >> ollama show model_name
# remove model >> ollama rm model_name

import ollama

def get_ollama_response(query, context, model):
    models = {
                'Llama4': 'llama4:latest', 
                'Llama3.1': 'llama3.1:8b', 
                'Llama3.2': 'llama3.2:3b',
                'Llama3': 'llama3:8b',
                'Gemma3': 'gemma3:latest', 
                'Gemma2': 'gemma2:9b',
                'Deepseek': 'deepseek-r1:8b', 
                'Qwen': 'qwen3:8b', 
                'Mistral': 'mistral:7b',
                'Vicuna': 'vicuna:7b',
                'Phi-4': 'phi4:14b'
              }
    
    response = ollama.chat(
        model=models[model],
        messages=[
            {"role": "system",
            "content": """
            You are a helpful assistant.
            You are provided with context and based on that you have to answer user question.
            Do not add any other information that is not mentioned in the context.
            """
            },
            
            {"role": "user", "content": f"""
            context: "{context}"
            question: "{query}" """
            },
        ],)
    return response["message"]["content"]


# %%
############################################ HuggingFace Endpoint ###########################################
# https://python.langchain.com/docs/integrations/llms/huggingface_endpoint/
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def get_huggingface_response(query, context, model):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    models = {
                'Llama': 'meta-llama/Llama-3.3-70B-Instruct',
                'Deepseek': "deepseek-ai/DeepSeek-R1-0528", 
                'Qwen': 'Qwen/Qwen2-7B-Instruct', 
                'Mistral': "mistralai/Mistral-Nemo-Base-2407"
              }

    # repos-
    # https://huggingface.co/mistralai/models
    # https://huggingface.co/deepseek-ai
    # https://huggingface.co/meta-llama/models
    # https://huggingface.co/Qwen

    llm = HuggingFaceEndpoint(
                        # endpoint_url="http://localhost:8010/",
                        repo_id=models[model], # 
                        task="conversational",
                        max_new_tokens=512,
                        # do_sample=False,
                        # top_k=10,
                        # top_p=0.95,
                        # typical_p=0.95,
                        # temperature=0.01,
                        # repetition_penalty=1.03,
                        provider="auto",  # let Hugging Face choose the best provider for you >> "auto"
                    )


    messages = [
            ("system", """You are a helpful assistant.
                You are provided with context and based on that you have to answer user question.
                Do not add any other information that is not mentioned in the context."""),
            ("user", f"""
                 context: "{context}"
                 question: "{query}" """)
        ]
    
    chat_model = ChatHuggingFace(llm=llm)

    return chat_model.invoke(messages).content

# response = get_huggingface_response(query='who is Mike ?', context='Mike is a doctor based out of colorado.', model='Llama')
# print(response)

# %%
def get_question_answer(query, context, framework, model):
    if framework == 'Groq':
        answer = get_groq_response(query, context, model)
    if framework == 'OpenAI':
        answer = get_opanai_response(query, context, model)
    if framework == 'Ollama':
        answer = get_ollama_response(query, context, model)
    if framework == 'HuggingFaceEndpoint':
        answer = get_huggingface_response(query, context, model)

    return answer
