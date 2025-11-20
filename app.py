import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------------------------------------
# TÍTULO
# -----------------------------------------------------------
st.title("📚 RAG Multi-PDF Inteligente – LangChain + Streamlit 🚀")

# -----------------------------------------------------------
# LLM (OpenAI)
# -----------------------------------------------------------
api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------------------------------------
# EMBEDDINGS (HuggingFace - Grátis)
# -----------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Inicializa storage do FAISS se ainda não existir
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------------------------------------------
# UPLOAD DE PDFs
# -----------------------------------------------------------
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

        st.success(f"📄 {uploaded.name} carregado com sucesso!")

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        docs = splitter.split_documents(docs)

        # Salva o nome do PDF em cada chunk
        for d in docs:
            d.metadata["pdf_name"] = uploaded.name

        all_docs.extend(docs)

    # Cria ou atualiza índice FAISS
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(all_docs, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_docs)

    st.success("✨ Índice atualizado com os novos PDFs!")

# -----------------------------------------------------------
# PERGUNTA DO USUÁRIO
# -----------------------------------------------------------
pergunta = st.text_input("🔎 Pergunte algo sobre os PDFs:")

if st.button("Enviar pergunta"):
    if st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado ainda.")
    elif not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        # Consulta vetorial
        docs = st.session_state.vectorstore.similarity_search(pergunta, k=6)

        # Construir o contexto
        contexto = ""
        for d in docs:
            contexto += f"\n\n[PDF: {d.metadata.get('pdf_name', 'desconhecido')}] ---\n{d.page_content}"

        prompt = f"""
Use SOMENTE o contexto abaixo para responder de forma clara.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])

        # Mostrar resposta
        st.subheader("🧠 Resposta:")
        st.write(resposta.content)

        # -----------------------------------------------------------
        # TRECHOS USADOS (SEM REPETIÇÕES)
        # -----------------------------------------------------------
        st.markdown("---")
        st.subheader("📌 Trechos usados:")

        shown = set()

        for d in docs:
            trecho = d.page_content.strip()

            # Remover cabeçalhos repetidos
            if trecho.startswith("Este documento pode ser verificado pelo código"):
                continue

            # Detectar duplicatas
            trecho_comp = trecho.replace("\n", " ").replace("  ", " ")[:300]
            if trecho_comp in shown:
                continue

            shown.add(trecho_comp)

            st.write(f"📄 **{d.metadata.get('pdf_name')}**")
            st.write(trecho[:500] + "...")
