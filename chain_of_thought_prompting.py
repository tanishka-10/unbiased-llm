# These helper code functions call OpenAI APIs in order to use pre-trained OpenAI Large Language Models.

!pip install openai
!pip install datasets
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.test.utils import common_texts
import nltk
import requests
import openai as ai
from datasets import load_dataset
import pandas as pd
import json
from google.colab import data_table

# will need to get own API key.


API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

# Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)


# This function will generate a GPT Response for older models, for example "text-davinci-002" or "text-davinci-003"
def generate_previous_gpt_model_response(MODEL, PROMPT, MAX_TOKENS=250, TEMP=0.99, TOP_P=0.5, N=1, FREQ_PEN=0.3,
                                         PRES_PEN=0.9):
    response = ai.Completion.create(
        engine=MODEL,
        # engine="text-davinci-002", # OpenAI has made four text completion engines available, named davinci, ada, babbage and curie. We are using davinci, which is the most capable of the four.
        prompt=PROMPT,  # The text file we use as input (step 3)
        max_tokens=MAX_TOKENS,  # how many maximum characters the text will consists of.
        temperature=TEMP,
        # temperature=int(temperature), # a number between 0 and 1 that determines how many creative risks the engine takes when generating text.,
        top_p=TOP_P,  # an alternative way to control the originality and creativity of the generated text.
        n=N,  # number of predictions to generate
        frequency_penalty=FREQ_PEN,
        # a number between 0 and 1. The higher this value the model will make a bigger effort in not repeating itself.
        presence_penalty=PRES_PEN
        # a number between 0 and 1. The higher this value the model will make a bigger effort in talking about new topics.
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


# Load Bias Dataset
dataset = load_dataset("md_gender_bias", "convai2_inferred")
prompt = """
The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.
The odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.
The odd numbers in this group add up to an even number: 16,  11, 14, 4, 8, 13, 24.
A: Adding all the odd numbers (11, 13) gives 24. The answer is True.
The odd numbers in this group add up to an even number: 17,  9, 10, 12, 13, 4, 2.
A: Adding all the odd numbers (17, 9, 13) gives 39. The answer is False.
The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:
"""
model="gpt-3.5-turbo"
generate_newer_gpt_model_response(model, prompt)
