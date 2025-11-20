import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.title("RAG + LangChain + Streamlit + GitHub 📄🚀")

# Chave da API (via Streamlit Secrets)
api_key = st.secrets["OPENAI_API_KEY"]

# Modelo LLM da OpenAI
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# Modelo de Embeddings NOVO e barato (evita RateLimit)
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

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

    # 🔥 Dividir o PDF em chunks menores (EVITA RATE LIMIT)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(documents)

    # Criar banco vetorial
    vectorstore = FAISS.from_documents(documents, embeddings)

    st.info("PDF indexado! Agora você pode fazer perguntas sobre ele.")

    # Pergunta do usuário
    pergunta = st.text_input("Pergunte algo sobre o PDF:")

    if st.button("Enviar pergunta"):
        if pergunta:
            # Buscar trechos mais relevantes
            docs = vectorstore.similarity_search(pergunta, k=3)

            # Criar o contexto
            contexto = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            Use o contexto abaixo para responder a pergunta do usuário:

            CONTEXTO:
            {contexto}

            PERGUNTA:
            {pergunta}
            """

            # Gerar resposta
            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.write(resposta.content)
        else:
            st.warning("Digite uma pergunta antes de enviar.")
