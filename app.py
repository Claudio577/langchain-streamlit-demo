import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG + LangChain + Streamlit + GitHub 📄🚀")

api_key = st.secrets["OPENAI_API_KEY"]

# LLM (continua usando OpenAI)
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# EMBEDDINGS LOCAIS (NÃO usam OpenAI – sem limite)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

uploaded_file = st.file_uploader("Envie um PDF para análise:", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF carregado com sucesso!")

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(documents, embeddings)

    st.info("PDF indexado! Faça uma pergunta:")

    pergunta = st.text_input("Pergunte algo sobre o PDF:")

    if st.button("Enviar pergunta"):
        docs = vectorstore.similarity_search(pergunta, k=3)
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
