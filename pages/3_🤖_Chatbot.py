import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

# lotties
import json
import requests
from streamlit_lottie import st_lottie


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


st.set_page_config(page_title="ML APP - Chatbot")

st.title("🤗Machine Learning Assistant Chatbot💬")

st.info("The place where you can answer your with me!")

lottie_code = load_lottiefile("lottiefiles/chatbot.json")

with st.sidebar:
    st_lottie(lottie_code, speed=1, loop=True, quality="low")

# Generate empty lists for generated and past.
## generated stores AI generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = ["I'm DAP, How may I help you?"]
## past stores User's questions
if "past" not in st.session_state:
    st.session_state["past"] = ["Hi!"]

# Layout of input/response containers
input_container = st.container()
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = hugchat.ChatBot(cookie_path="cookie_path.json")
    response = chatbot.chat(prompt)
    return response


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
