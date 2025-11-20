import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


st.title("RAG Multi-PDF + FAISS Incremental + LangChain + Streamlit 🚀")


# ============================
# 1) Carregar chave da API
# ============================
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.2
)

# ============================
# 2) Embeddings com cache
# ============================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()


# ============================
# 3) Estado global do FAISS
# ============================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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

        # Adicionar metadado do nome do pdf
        for c in chunks:
            c.metadata["pdf_name"] = uploaded_file.name

        all_docs.extend(chunks)

    return all_docs


# ============================
# 5) Adicionar ao índice FAISS
# ============================
def add_to_vectorstore(docs, embeddings):
    # Se não existe índice, criamos
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        # Se já existe, só adicionamos novos docs
        st.session_state.vectorstore.add_documents(docs)


# ============================
# 6) Upload de PDFs
# ============================
uploaded_files = st.file_uploader(
    "Envie um ou vários PDFs:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info("Processando PDFs...")

    docs = process_pdfs(uploaded_files)

    st.info("Adicionando ao índice vetorial...")
    add_to_vectorstore(docs, embeddings)

    st.success("PDFs adicionados ao índice com sucesso!")


# ============================
# 7) Campo de Pergunta
# ============================
pergunta = st.text_input("Faça sua pergunta:")

if st.button("Enviar pergunta"):
    if not pergunta:
        st.warning("Digite uma pergunta.")
    elif st.session_state.vectorstore is None:
        st.error("Nenhum PDF foi carregado ainda!")
    else:
        docs = st.session_state.vectorstore.similarity_search(pergunta, k=5)

        contexto = ""
        for d in docs:
            contexto += f"\n\n--- [PDF: {d.metadata.get('pdf_name')}] ---\n{d.page_content}"

        prompt = f"""
Responda APENAS com base no contexto abaixo. 
Se a resposta não estiver nos PDFs, diga isso claramente.

### CONTEXTO:
{contexto}

### PERGUNTA:
{pergunta}

### RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])
        st.write(resposta.content)
