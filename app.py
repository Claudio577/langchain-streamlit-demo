import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

st.title("Demo LangChain + Streamlit + GitHub 🚀")

# Chave vem do Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini"
)

user_input = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if user_input:
        resposta = llm([HumanMessage(content=user_input)])
        st.write(resposta.content)
    else:
        st.warning("Digite algo antes de enviar!")
