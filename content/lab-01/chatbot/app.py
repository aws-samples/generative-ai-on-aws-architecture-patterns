# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import uuid
import sys
import json
import os
import requests
import uuid

AI_ICON = "app_icon.png"
base_url = os.getenv('BASE_URL')
headers = {'Content-Type': 'application/json'}

st.set_page_config(page_title="LLM-powered chatbot")

# Sidebar contents
with st.sidebar:
    st.title('💬 Chatbot App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
    - [Amazon Bedrock](https://aws.amazon.com/bedrock/)
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/docs/get_started/introduction)
    ''')
    add_vertical_space(5)
    st.write('Made with ❤ for AWS workshops')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 32px;
                    padding-bottom: 32px;
                    padding-left: 0;
                    padding-right: 0;
                }
                .element-container img {
                    background-color: #000000;
                }

                .main-header {
                    font-size: 16px;
                }
        </style>
        """, unsafe_allow_html=True)


def create_session_id():
    return str(uuid.uuid4())

# Create or get the session state
def get_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = create_session_id()
    return st.session_state.session_id

session_id = get_session()

# Refresh button callback
def refresh():
    session_id = create_session_id()
    st.session_state.session_id = session_id
    st.session_state['generated'] = ["Hi, I'm LLM chatbot. How may I help you?"]
    st.session_state['past'] = []


def clear():
    st.session_state.session_id = session_id
    st.session_state['generated'] = ["Hi, I'm LLM bot. How may I help you?"]
    st.session_state['past'] = []
    st.session_state['input'] = ""


def write_logo():
    col1, col2, col3 = st.columns([5, 1, 5])
    with col2:
        st.image(AI_ICON, use_column_width='always')


def write_top_bar():
    col1, col2, col3, col4, col5 = st.columns([1,6,2,2,2])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        st.write(f"<h4 class='main-header'>LLM Chatbot</h4>",  unsafe_allow_html=True)
    with col5:
        bedrock = st.checkbox(label='Use Bedrock', value=True)
        st.session_state['USE_BEDROCK'] = bedrock
    with col3:
        if st.button("Clear Chat", key="clear"):
            clear()
    with col4:
        if st.button('Reset Session'):
            refresh()

write_top_bar()

session_header = f" Session ID:  {session_id}"
st.write(f"<h4 class='main-header'>{session_header}</h4>",  unsafe_allow_html=True)

colored_header(label='', description='', color_name='blue-30')

# Layout of input/response containers
input_container = st.container()
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("User Input: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Generate empty lists for generated and past.
# generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hi, I'm LLM bot. How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    url = f'{base_url}/backendapp'
    use_bedrock = st.session_state['USE_BEDROCK']
    body = {"query": prompt, "uuid": session_id, "USE_BEDROCK": use_bedrock}
    response = requests.post(url, headers=headers, data=json.dumps(body), verify=False)
    output_text = response.text
    return output_text

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            if i > 0:
                message(st.session_state['past'][i-1], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
