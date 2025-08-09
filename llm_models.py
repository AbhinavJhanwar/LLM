# %%

############################################### LLaMa 3.2 ####################################################
# Text Encoder- Llama3.1
# Image Encoder- ViT-H/14
# Notes- trains image encoder and freeze llm model to be able to utlize the same image encoder with
# other 11b & 90b models without training.

"""

# https://huggingface.co/meta-llama
# https://huggingface.co/meta-llama/Llama-3.2-11B-Vision
# https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
# model weights- huggingface-cli download meta-llama/Llama-3.2-11B-Vision --include "original/*" --local-dir Llama-3.2-11B-Vision
# gpu required to run in local environment

## USING HUGGINGFACE
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
"""


# %%
################################################ Qwen.5-VL ###############################################
# Text Encoder- Qwen2
# Image Encoder- ?

"""

# https://github.com/QwenLM/Qwen2.5-VL
# https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5
# https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

## USING HUGGINGFACE
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

"""


# %%
############################################### LLaVa ####################################################
# Text Encoder- Llama3.1 & Mistral
# Image Encoder- CLIP ViT-L/14
# Notes- trains on mistral as well as llama models

"""

# https://huggingface.co/docs/transformers/model_doc/llava

# HUGGINGFACE
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Load the model in half-precision
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)

"""

# %%
############################################### MolMo ####################################################
# Text Encoder- Qwen2 72B and other dimensions
# Image Encoder- CLIP ViT-L/14
# Notes- trained on pixmo dataset


"""
# https://huggingface.co/allenai/Molmo-7B-D-0924
# https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19

# HUGGINGFACE
pip install einops torchvision
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# process the image and text
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe this image."
)

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)

# >>>  This image features an adorable black Labrador puppy, captured from a top-down
#      perspective. The puppy is sitting on a wooden deck, which is composed ...


"""


# %%
############################################### NVLM by Nvidia ####################################################
# Text Encoder- Qwen2-72B-Instruct
# Image Encoder- Intern ViT-6B
# Notes- trained on both unified embedding decoder (NVLD) & cross modality attention (NVLX) along with hybrid architecture (NVLH).

"""
# https://huggingface.co/nvidia/NVLM-D-72B

"""


# %%
############################################### Pixtral by Mistral ####################################################
# Text Encoder- Mistral NeMo
# Image Encoder- ViT trained from scratch
# Notes- trained on unified embedding decoder architecture


"""
# https://huggingface.co/mistralai/Pixtral-12B-2409
# https://huggingface.co/docs/transformers/main/en/model_doc/pixtral

# HUGGINGFACE
pip install --upgrade vllm
pip install --upgrade mistral_common

from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

prompt = "Describe this image in one sentence."
image_url = "https://picsum.photos/id/237/200/300"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
    },
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)

"""
# %%
############################################### Ollama ####################################################
# https://ollama.com/download
# https://ollama.com/library
# famouse vision models-
# llama3.2-vision:Xb, replace X with 11b, 90b
# llava:Xb, 7b, 13b, 34b
# qwen2.5vl:Xb, 3b, 7b, 32b, 72b
# minicpm-v
# llava-llama3
# llama4

# download model >> ollama pull model_name
# list all downloaded models >> ollama list
# check model properties >> ollama show model_name
# remove model >> ollama rm model_name

import ollama

def get_ollama_image_response(image_path):
    response = ollama.chat(
        model="qwen2.5vl:3b",
        # model='llava:7b',
        messages=[
            {
                "role": "user", 
                "content": "Give a summary of the image provided. Be descriptive.",
                "images":[f'{image_path}']
                # "content":"hello"
            },
        ],
    )
    return response["message"]["content"]
'''
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

llm.invoke("tell me about partial functions in python")
'''


# %%
############################################### Ollama async ####################################################
import asyncio
from ollama import AsyncClient

async def chat():
    """
    Stream a chat from Llama using the AsyncClient.
    """
    message = {
        "role": "user",
        "content": "hello"
    }
    async for part in await AsyncClient().chat(
        model="qwen2.5vl:3b", messages=[message], stream=True
    ):
        print(part["message"]["content"], end="", flush=True)


# asyncio.run(chat())
# await chat() # if running in interactive window

# %%
############################################### Groq ####################################################
from groq import Groq
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# image_path = 'pymupdf_images/1506.01497v3.pdf-9-2.png'

def get_groq_response(image_path):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # Getting the base64 string
    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
                "role": "user",
                "content": 
                [
                    {
                        "type": "text", 
                        "text": "Give a summary of the image provided. Be descriptive"
                    },
                    {
                        "type": "image_url",
                        "image_url": 
                        {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
    )

    return completion.choices[0].message.content
# %%
############################################### ChatGPT ###############################################
from dotenv import load_dotenv
import os
import base64
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_chat_gpt_response(image_path):
    # Access environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    client = OpenAI()

    # Getting the Base64 string
    base64_image = encode_image(image_path)

    msg = client.responses.create(
        model="gpt-4.1",
        input=[{
                "role":"user",
                "content": 
                [
                        {
                            "type": "input_text", 
                            "text" : "Give a summary of the image provided. Be descriptive"
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        },
                ],
                }])
    return msg.output_text

