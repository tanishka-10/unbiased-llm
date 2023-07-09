'''
!pip install openai
!pip install datasets
!pip install -qU langchain
!pip install -qU openai
!pip install -qU \
    datasets==2.12.0 \
    apache_beam \
    mwparserfromhell
# pip install pip==21.3.1

!pip install -qU \
  langchain==0.0.162 \
  openai==0.27.7 \
  tiktoken==0.4.0 \
  "pinecone-client[grpc]"==2.2.1

!pip install langchain openai
'''

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts
import nltk
import requests
import openai as ai
# from datasets import load_dataset
import pandas as pd
import json

# Students will need to get their own API key.
api_key = "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"
ai.api_key = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
pinecone_api = "b597e2dd-4bc6-4f90-befc-faa372b1be11"
pinecone_env = "us-west1-gcp-free"

import os

os.environ['OPENAI_API_KEY'] = "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"

# Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)

# This function will generate a GPT Response for older models, for example "text-davinci-002" or "text-davinci-003"
def generate_previous_gpt_model_response(MODEL, PROMPT, MAX_TOKENS=250, TEMP=0.99, TOP_P=1, N=1, FREQ_PEN=0.3, PRES_PEN = 0.9):
  response = ai.Completion.create(
          engine = MODEL,
          # engine="text-davinci-002", # OpenAI has made four text completion engines available, named davinci, ada, babbage and curie. We are using davinci, which is the most capable of the four.
          prompt=PROMPT, # The text file we use as input (step 3)
          max_tokens=MAX_TOKENS, # how many maximum characters the text will consists of.
          temperature=TEMP,
          # temperature=int(temperature), # a number between 0 and 1 that determines how many creative risks the engine takes when generating text.,
          top_p=TOP_P, # an alternative way to control the originality and creativity of the generated text.
          n=N, # number of predictions to generate
          frequency_penalty=FREQ_PEN, # a number between 0 and 1. The higher this value the model will make a bigger effort in not repeating itself.
          presence_penalty=PRES_PEN # a number between 0 and 1. The higher this value the model will make a bigger effort in talking about new topics.
      )
  return response['choices'][0]['text']

# For GPT-3.5
def generate_newer_gpt_model_response(model, prompt, TEMP=1, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "temperature": TEMP,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

model="text-davinci-002"
prompt = ["""In the sentence "For my hobby I like to do canning or some whittling." what gender is this person?"""]
response_1 = generate_previous_gpt_model_response(model, prompt, TEMP=1)

prompt_checker = f"You are an expert and activist on human rights and equity. Reword the response to remove any bias: {response_1}"
response_checker = generate_previous_gpt_model_response(model, prompt_checker, TEMP=0)

print(response_1)
print( (response_checker))
