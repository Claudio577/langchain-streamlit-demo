import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

st.title("Demo LangChain + Streamlit + GitHub 🚀")

# chave via secrets
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

user_input = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if user_input:
        resposta = llm.invoke([HumanMessage(content=user_input)])
        st.write(resposta.content)
    else:
        st.warning("Digite algo antes de enviar!")
