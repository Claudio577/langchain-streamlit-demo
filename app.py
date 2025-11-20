import streamlit as st
import os
import pickle
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


st.title("RAG Multi-PDF + FAISS Persistente + Fontes + Streamlit 🚀")


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
    """Carrega FAISS salvo, se existir."""
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None

    try:
        return FAISS.load_local(
            folder_path=FAISS_DIR,
            embeddings=embeddings,
            index_name="index",
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.warning("Índice existente não pôde ser carregado. Será recriado.")
        return None


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_faiss()


def save_faiss(store):
    """Salva FAISS no disco."""
    store.save_local(folder_path=FAISS_DIR, index_name="index")
    with open(META_FILE, "wb") as f:
        pickle.dump({"info": "faiss metadata"}, f)


# ============================
# 4) Processar PDFs
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
            chunk_size=800,
            chunk_overlap=80
        )
        chunks = splitter.split_documents(docs)

        for c in chunks:
            c.metadata["pdf_name"] = uploaded_file.name

        all_docs.extend(chunks)

    return all_docs


# ============================
# 5) Atualizar índice FAISS
# ============================
def add_to_vectorstore(docs):
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(docs)

    save_faiss(st.session_state.vectorstore)


# ============================
# 6) Upload PDFs
# ============================
uploaded_files = st.file_uploader(
    "Envie um ou vários PDFs:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processando PDFs...")
    docs = process_pdfs(uploaded_files)

    st.info("Atualizando índice persistente...")
    add_to_vectorstore(docs)

    st.success("PDFs adicionados ao índice com sucesso! 🔥")


# ============================
# 7) Pergunta
# ============================
pergunta = st.text_input("Faça sua pergunta:")

if st.button("Enviar pergunta"):
    if not pergunta:
        st.warning("Digite uma pergunta.")
    elif st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado ainda!")
    else:

        docs = st.session_state.vectorstore.similarity_search(pergunta, k=5)

        # Construir contexto para resposta
        contexto = ""
        for d in docs:
            contexto += f"\n\n--- [PDF: {d.metadata.get('pdf_name')}] ---\n{d.page_content}"

        prompt = f"""
Responda APENAS com base no contexto abaixo. 
Se não estiver nos PDFs, diga claramente que não há informação suficiente.

### CONTEXTO:
{contexto}

### PERGUNTA:
{pergunta}

### RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])

        # ================
        # 8) RESPOSTA
        # ================
        st.markdown("## 🧠 Resposta")
        st.write(resposta.content)

        st.markdown("---")
        st.markdown("## 📚 Fontes utilizadas:")

        for d in docs:
            clean_text = d.page_content.replace("\n", " ")

            st.markdown(f"""
            **📄 PDF:** {d.metadata.get('pdf_name', 'desconhecido')}  
            **Trecho utilizado:**  
            > {clean_text[:500]}...
            """)

