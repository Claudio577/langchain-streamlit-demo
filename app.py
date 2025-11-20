import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

st.title("RAG Multi-PDF + LangChain + FAISS + Streamlit 📚🚀")

api_key = st.secrets["OPENAI_API_KEY"]

# -----------------------------
# 1) LLM OpenAI
# -----------------------------
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.2,
)


# -----------------------------
# 2) Embeddings locais (rápido & gratuito)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# -----------------------------
# 3) Função para processar PDFs
# -----------------------------
@st.cache_resource
def process_pdfs(files):
    all_docs = []

    for uploaded_file in files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80
        )
        chunks = splitter.split_documents(docs)

        # registrar nome do PDF
        for c in chunks:
            c.metadata["pdf_name"] = uploaded_file.name

        all_docs.extend(chunks)

    return all_docs


# -----------------------------
# 4) Upload de PDFs
# -----------------------------
uploaded_files = st.file_uploader(
    "Envie um ou vários PDFs:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success("PDFs carregados! Processando...")

    # Processar
    docs = process_pdfs(uploaded_files)

    # Montar FAISS
    st.info("Criando índice vetorial...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("Índice criado! Agora faça sua pergunta abaixo ⬇️")

    pergunta = st.text_input("Pergunta sobre os PDFs:")

    if pergunta:
        if st.button("Enviar pergunta"):

            # Recuperação via FAISS
            retrieved_docs = vectorstore.similarity_search(pergunta, k=5)

            # Montar contexto
            contexto = ""
            for d in retrieved_docs:
                contexto += (
                    f"\n\n--- [PDF: {d.metadata.get('pdf_name')}] ---\n"
                    f"{d.page_content}"
                )

            prompt = f"""
Você é um assistente especializado em análise de documentos.

Responda APENAS com base nas informações encontradas nos PDFs.
Se não encontrar a resposta nos documentos, diga explicitamente que não há informação suficiente.

### CONTEXTO:
{contexto}

### PERGUNTA:
{pergunta}

### RESPOSTA:
"""

            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.write(resposta.content)

