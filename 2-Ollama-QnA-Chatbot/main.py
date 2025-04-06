import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries"),
        ("user", "Question: {question}"),
    ]
)

def generate_response(
    question: str,
    model_name: str,
    temperature: float,
) -> str:
    """
    Generate a response to a user's question using the given Ollama model.

    Args:
        question (str): The user's question.
        model_name (str): The name of the Ollama model to use.
        temperature (float): The temperature to use for generation.

    Returns:
        str: The generated response to the user's question.
    """
    llm = Ollama(model=model_name, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Title of the app
st.title("Q&A Chatbot with Ollama")

# Sidebar for settings
st.sidebar.title("Settings")

# Dropdown to select various Ollama models
model_name = st.sidebar.selectbox("Select an Ollama Model", ["gemma3:1b", "llama3:latest"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name, temperature)
    st.write(response)
else:
    st.write("Please provide the query")
