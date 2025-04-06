import streamlit as st
import openai 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(
    question: str,
    api_key: str,
    llm: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Generate a response to a user's question using the gpt-3.5-turbo model.

    Args:
        question (str): The user's question.
        api_key (str): The API key for the OpenAI API.
        llm (str): The name of the language model to use.
        temperature (float): The temperature to use for generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The generated response to the user's question.
    """
    llm = ChatOpenAI(temperature=temperature, model_name=llm, max_completion_tokens=max_tokens, openai_api_key=api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

## Title of the app
st.title("Q&A Chatbot With OpenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key: ", type="password")

## Dropdown to select various OpenAI models
llm = st.sidebar.selectbox("Select an OPenAI Model", ["gpt-4o","gpt-4o-mini","gpt-4","gpt-4-turbo"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")