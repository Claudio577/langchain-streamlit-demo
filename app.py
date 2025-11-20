import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG Multi-PDF Contínuo 📚🚀")

# ============ LLM ============
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# ============ Embeddings ============
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Criar vetor na sessão
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ============ Upload ============

uploaded_files = st.file_uploader(
    "Envie PDFs (pode enviar novos a qualquer momento):",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    for uploaded in uploaded_files:
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Ler PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        docs = splitter.split_documents(docs)

        # Guardar nome do PDF
        for d in docs:
            d.metadata["pdf_name"] = uploaded.name

        all_docs.extend(docs)

        st.success(f"PDF carregado: {uploaded.name}")

    # Atualizar o FAISS existente ou criar um novo
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(all_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_docs)

    st.success("Índice atualizado com novos PDFs! 🔥")

# ============ Pergunta ============

pergunta = st.text_input("Pergunte algo sobre os PDFs carregados:")

if st.button("Enviar pergunta"):
    if not st.session_state.vectorstore:
        st.error("Nenhum PDF carregado ainda.")
    elif not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        docs = st.session_state.vectorstore.similarity_search(pergunta, k=4)

        contexto = ""
        for d in docs:
            contexto += f"\n\n[PDF: {d.metadata.get('pdf_name', 'desconhecido')}] ---\n{d.page_content}"

        prompt = f"""
Use SOMENTE o contexto abaixo para responder.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])
        st.write(resposta.content)

        st.markdown("---")
        st.subheader("Trechos usados:")

        for d in docs:
            st.write(f"📄 **{d.metadata.get('pdf_name')}**")
            st.write(d.page_content[:400] + "...")
