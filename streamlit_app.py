import streamlit as st
import openai as ai


## Use your own API key: https://platform.openai.com/account/api-keys



def chatgpt_call(prompt, model, temperature):
  completion = ai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=temperature
  )
  return completion['choices'][0]['message']['content']

st.header('Removing Bias')
topic = st.text_input('Topic you want to learn')
model = 'gpt-3.5-turbo' # "gpt-3.5-turbo"
temperature = 1
st.sidebar.markdown("This app uses OpenAI's generative AI. Please use it carefully and check any output as it could be biased or wrong. ")

prompt = f"Explain this concept to me as if I am 5 years old: {topic}"

explanation = chatgpt_call(prompt, model, temperature)
temperature = 0
prompt_2 = f"You are an expert in human rights. Reword this response so no discrimination towards gender is included: {explanation}"
explanation_2 = chatgpt_call(prompt_2, model, temperature)
generate = st.button('Generate Response')

if generate:
  st.markdown(explanation)
  st.balloons()
