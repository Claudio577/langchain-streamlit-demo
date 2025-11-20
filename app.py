import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("RAG + LangChain + Streamlit + GitHub 📄🚀")

# Chave da API
api_key = st.secrets["OPENAI_API_KEY"]

# Modelo LLM
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# Embeddings novos
embeddings = OpenAIEmbeddings(
    api_key=api_key,
    model="text-embedding-3-small"
)

# Upload de arquivo
uploaded_file = st.file_uploader("Envie um PDF para análise:", type=["pdf"])

if uploaded_file:
    # Salvar arquivo temporário
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF carregado com sucesso!")

    # Carregar PDF e dividir em chunks
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(documents)

    # Criar banco vetorial
    vectorstore = FAISS.from_documents(documents, embeddings)

    st.info("PDF indexado! Faça uma pergunta:")

    pergunta = st.text_input("Pergunte algo sobre o PDF:")

    if st.button("Enviar pergunta"):
        if pergunta:
            docs = vectorstore.similarity_search(pergunta, k=3)

            contexto = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            Use o contexto abaixo para responder a pergunta do usuário:

            CONTEXTO:
            {contexto}

            PERGUNTA:
            {pergunta}
            """

            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.write(resposta.content)
        else:
            st.warning("Digite uma pergunta.")
