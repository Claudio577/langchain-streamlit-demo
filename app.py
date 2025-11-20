import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

st.title("RAG + LangChain + Streamlit + GitHub 📄🚀")

api_key = st.secrets["OPENAI_API_KEY"]

# Modelo LLM
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# Embeddings
embeddings = OpenAIEmbeddings(api_key=api_key)

# Upload do PDF
uploaded_file = st.file_uploader("Envie um PDF para análise:", type=["pdf"])

if uploaded_file:
    # Salvar arquivo temporário
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF carregado com sucesso!")

    # Carregar PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Criar vetorstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    st.info("PDF indexado! Agora você pode fazer perguntas sobre ele.")

    # Campo de pergunta
    pergunta = st.text_input("Pergunte algo sobre o PDF:")

    if st.button("Enviar pergunta"):
        if pergunta:
            # Busca contextual
            docs = vectorstore.similarity_search(pergunta, k=3)

            # Monta prompt com contexto
            contexto = "\n\n".join([d.page_content for d in docs])
            prompt = f"""
            Use o contexto abaixo para responder:

            CONTEXTO:
            {contexto}

            PERGUNTA:
            {pergunta}
            """

            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.write(resposta.content)
        else:
            st.warning("Digite uma pergunta antes de enviar.")
