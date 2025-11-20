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


# ============================
# MODO 3 — PERGUNTAS (RAG)
# ============================
else:
    # 1) Tentar busca normal
    docs = st.session_state.vectorstore.similarity_search(pergunta, k=5)

    # 2) Se a busca não retornar nada → fallback com busca mais ampla
    if len(docs) == 0:
        st.warning("Nenhum trecho relevante encontrado — buscando trecho geral do PDF.")
        docs = st.session_state.vectorstore.similarity_search("", k=20)

    # 3) Se ainda assim não houver trechos → erro amigável
    if len(docs) == 0:
        st.error("Não foi possível recuperar informações do PDF. Tente outra pergunta.")
        st.stop()

    # 4) Monta contexto
    contexto = ""
    for d in docs:
        contexto += f"\n\n---[PDF: {d.metadata.get('pdf_name')}]---\n{d.page_content}"

    # 5) Prompt mais inteligente, sem bloquear resposta
    prompt = f"""
Use o contexto abaixo como fonte principal. 
Se algum trecho estiver incompleto, responda da forma mais útil possível.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA (clara, direta e completa):
"""

    resposta = llm.invoke([HumanMessage(content=prompt)])

    st.subheader("🧠 Resposta")
    st.write(resposta.content)

    # Mostrar fontes usadas
    st.markdown("---")
    st.subheader("📚 Fontes usadas:")

    pdf_groups = {}
    for d in docs:
        pdf_name = d.metadata.get("pdf_name", "desconhecido")
        if pdf_name not in pdf_groups:
            pdf_groups[pdf_name] = d

    for pdf_name, d in pdf_groups.items():
        clean = d.page_content.replace("\n", " ")
        st.markdown(f"""
        **📄 PDF:** {pdf_name}  
        > {clean[:500]}...
        """)


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

