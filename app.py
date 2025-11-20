import streamlit as st
import os
import pickle
import shutil
import uuid

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


st.title("📚 RAG Inteligente PRO — Multi-PDF + Resumo + Explicação + Busca 🔍")


# ============================
# 1) LLM
# ============================
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.2
)


# ============================
# 2) Embeddings
# ============================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# ============================
# 3) Persistência FAISS
# ============================
FAISS_DIR = "faiss_index"
os.makedirs(FAISS_DIR, exist_ok=True)

INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
META_FILE = os.path.join(FAISS_DIR, "index.pkl")


def load_faiss():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None

    try:
        return FAISS.load_local(
            folder_path=FAISS_DIR,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
    except Exception:
        return None


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_faiss()


def save_faiss(store):
    store.save_local(folder_path=FAISS_DIR, index_name="index")
    with open(META_FILE, "wb") as f:
        pickle.dump({"info": "faiss metadata"}, f)


# ============================
# 4) Botão de limpeza total
# ============================
st.markdown("### 🗑️ Limpar todos os PDFs e reiniciar índice")

if st.button("Apagar todos os PDFs e reiniciar índice"):
    try:
        shutil.rmtree(FAISS_DIR)
        os.makedirs(FAISS_DIR, exist_ok=True)

        st.session_state.vectorstore = None

        st.session_state.uploader_key = str(uuid.uuid4())

        st.success("Todos os PDFs foram apagados e o índice foi reiniciado!")
        st.rerun()

    except Exception as e:
        st.error(f"Erro ao limpar índice: {e}")


# ============================
# 5) Processar PDFs
# ============================
def process_pdfs(files):
    all_docs = []

    for uploaded_file in files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(docs)

        for c in chunks:
            c.metadata["pdf_name"] = uploaded_file.name

        all_docs.extend(chunks)

    return all_docs


# ============================
# 6) Atualizar FAISS
# ============================
def add_to_vectorstore(docs):
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(docs)

    save_faiss(st.session_state.vectorstore)


# ============================
# 7) Upload de PDFs
# ============================
uploaded_files = st.file_uploader(
    "Envie PDFs:",
    type=["pdf"],
    accept_multiple_files=True,
    key=st.session_state.get("uploader_key", "uploader_key")
)

if uploaded_files:
    st.info("Processando PDFs...")
    docs = process_pdfs(uploaded_files)

    st.info("Atualizando índice persistente...")
    add_to_vectorstore(docs)

    st.success("PDFs adicionados com sucesso! 🔥")


# ============================
# 8) UI — Modo Inteligente
# ============================
modo = st.radio(
    "Escolha o modo de resposta:",
    ["📄 Resumo do PDF", "💡 Explicar / Interpretar PDF", "🔍 Perguntas específicas (RAG)"]
)

pergunta = st.text_input("Digite sua pergunta:")

if st.button("Enviar pergunta"):
    if not pergunta:
        st.warning("Digite sua pergunta.")
    elif st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado.")
    else:

        # ============================
        # MODO 1 — RESUMO
        # ============================
        if modo == "📄 Resumo do PDF":
            docs = st.session_state.vectorstore.similarity_search("", k=20)
            full_text = " ".join([d.page_content for d in docs])

            prompt = f"""
Faça um RESUMO completo, estruturado e claro do documento abaixo:

DOCUMENTO:
{full_text}

RESUMO:
"""
            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.subheader("📄 Resumo do PDF")
            st.write(resposta.content)

        # ============================
        # MODO 2 — EXPLICAÇÃO
        # ============================
        elif modo == "💡 Explicar / Interpretar PDF":
            docs = st.session_state.vectorstore.similarity_search("", k=20)
            full_text = " ".join([d.page_content for d in docs])

            prompt = f"""
Explique o conteúdo do documento abaixo de forma simples, clara e organizada.
Depois responda à pergunta: {pergunta}

DOCUMENTO:
{full_text}

EXPLICAÇÃO:
"""
            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.subheader("💡 Explicação do PDF")
            st.write(resposta.content)

        # ============================
        # MODO 3 — PERGUNTAS (RAG)
        # ============================
        else:
            docs = st.session_state.vectorstore.similarity_search(pergunta, k=5)

            contexto = ""
            for d in docs:
                contexto += f"\n\n---[PDF: {d.metadata.get('pdf_name')} ]---\n{d.page_content}"

            prompt = f"""
Responda APENAS usando o contexto.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

            resposta = llm.invoke([HumanMessage(content=prompt)])

            st.subheader("🧠 Resposta")
            st.write(resposta.content)

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

pq essa resposta quando eu poerguntei o que o pdf fala

