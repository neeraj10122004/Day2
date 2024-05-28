import streamlit as st
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

st.title("Csv_Data_Analysis_LLM_Agent")

if api_key := st.text_input("Enter the API Key for OpenAi"):
    os.environ["OPENAI_API_KEY"] = api_key
    df = pd.read_csv("customers.csv")
    data_analysis_agent = create_pandas_dataframe_agent(OpenAI(temperature=0),df,verbose=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response=data_analysis_agent.invoke("Analyze this data.")
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})