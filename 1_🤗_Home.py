import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
import time

st.set_page_config(
    page_title="ML APP",
    page_icon="👋",
)


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_robot = load_lottiefile("lottiefiles/logo.json")

lottie_code = load_lottiefile("lottiefiles/robot_orange.json")
st.title("Hello World!")
st.header("This is an amazing AutoML app😊")

# sidebar decoration
with st.sidebar:
    st_lottie(lottie_robot, speed=1, loop=True, quality="low")
    st.success("Select a page above.")
st.info("You can try our services in the navigation")
st_lottie(lottie_code, speed=1, loop=True, quality="low")

# snowfall
if st.button(":pink[Funny Button]🤡"):
    with st.spinner("Wait for it..."):
        time.sleep(2)
    st.snow()
