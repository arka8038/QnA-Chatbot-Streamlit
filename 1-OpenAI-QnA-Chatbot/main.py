import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "Q&A Chatbot with OpenAI"

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries"),
        ("user", "Question: {question}")
    ]
)

def generate_response(
    question: str,
    openai_api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int
) -> str:
    """
    Generate a response to a user's question using the given OpenAI model.

    Args:
        question (str): The user's question.
        openai_api_key (str): The API key for the OpenAI API.
        model_name (str): The name of the language model to use.
        temperature (float): The temperature to use for generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str: The generated response to the user's question.
    """
    llm = ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        max_completion_tokens=max_tokens,
        openai_api_key=openai_api_key
    )
    output_parser = StrOutputParser()
    chain = PROMPT_TEMPLATE | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

st.title("Q&A Chatbot With OpenAI")

st.sidebar.title("Settings")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API key: ", type="password")

model_name = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o","gpt-4o-mini","gpt-4","gpt-4-turbo"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, openai_api_key, model_name, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")
