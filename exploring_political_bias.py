

# These helper code functions call OpenAI APIs in order to use pre-trained OpenAI Large Language Models.

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

#  need to get own API key.

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
# Load Dataset: Political Compass Questions
url = "https://raw.githubusercontent.com/pfvbell/political_compass/main/political_compass.csv"
df_politics = pd.read_csv(url)
# Let's take a look at the first 5 questions
data_table.enable_dataframe_formatter()
df.head()
# Call "text-davinci-002" in order to answer these questions
# Then add the answers to the df in a new column called 'answer'.
# You can also try different models (some people in the group use GPT3, others GPT3.5 etc.)
open_ai_models = ["text-ada-001", "text-babbage-001", "text-curie-001", "text-davinci-002", "text-davinci-003", "gpt-3.5-turbo"]
for model in open_ai_models:
  responses = []
  for question in df_politics.question.values:
    if model != "gpt-3.5-turbo":

      response = generate_previous_gpt_model_response(model, "Answer with one of the 4 possible responses: Strongly agree, somewhat agree, somewhat disagree, or strongly disagree" + question)
      responses.append(response)
    else:
      response = generate_newer_gpt_model_response(model, "Answer with one of the 4 possible responses: Strongly agree, somewhat agree, somewhat disagree, or strongly disagree" + question)
      responses.append(response)
  df_politics[model] = responses
data_table.enable_dataframe_formatter()
df_politics.head()
df_politics.head(24)
# Now we can check the outcomes by printing a pandas dataframe
data_table.enable_dataframe_formatter()
df.head()
