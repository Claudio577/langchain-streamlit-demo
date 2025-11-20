import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
def clean_text_block(text: str) -> str:
    """
    Normaliza textos quebrados em várias linhas, juntando palavras
    e removendo quebras estranhas vindas de PDFs.
    """
    lines = text.split("\n")
    new_lines = []
    buffer = ""

    for line in lines:
        line_strip = line.strip()

        # ignora linhas vazias
        if not line_strip:
            if buffer:
                new_lines.append(buffer)
                buffer = ""
            continue

        # se a linha é curta, provavelmente faz parte da mesma frase
        if len(line_strip.split()) <= 3:
            buffer += " " + line_strip
        else:
            # se a linha anterior não terminou frase, junta
            if buffer and not buffer.endswith((".", "!", "?", ";", ":")):
                buffer += " " + line_strip
            else:
                if buffer:
                    new_lines.append(buffer)
                buffer = line_strip

    if buffer:
        new_lines.append(buffer)

    return "\n".join(new_lines)

# -----------------------------------------------------------
# TÍTULO
# -----------------------------------------------------------
st.title("📚 RAG Multi-PDF Inteligente – Reinício Automático 🚀")

# -----------------------------------------------------------
# LLM
# -----------------------------------------------------------
api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)

# -----------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Inicializa área da sessão
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_list" not in st.session_state:
    st.session_state.pdf_list = []

# -----------------------------------------------------------
# BOTÃO PARA LIMPAR MEMÓRIA
# -----------------------------------------------------------
st.markdown("### 🧹 Limpar PDFs carregados")

if st.button("🔄 Resetar memória e apagar todos os PDFs"):
    st.session_state.vectorstore = None
    st.session_state.pdf_list = []
    st.success("Memória limpa! Nenhum PDF carregado.")
    st.rerun()

# -----------------------------------------------------------
# UPLOAD DE PDFs
# -----------------------------------------------------------
uploaded_files = st.file_uploader(
    "Envie PDFs (um ou vários). Sempre será criado um índice novo:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    # SEMPRE RESETAR O ÍNDICE QUANDO ENVIAR NOVOS PDFs
    st.session_state.vectorstore = None
    st.session_state.pdf_list = []

    for uploaded in uploaded_files:
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"📄 {uploaded.name} carregado!")

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        docs = splitter.split_documents(docs)

        for d in docs:
            d.metadata["pdf_name"] = uploaded.name

        all_docs.extend(docs)
        st.session_state.pdf_list.append(uploaded.name)

    # Criar novo índice FAISS
    st.session_state.vectorstore = FAISS.from_documents(all_docs, embeddings)
    st.success("✨ Novo índice criado! PDFs atuais prontos para perguntas.")

# -----------------------------------------------------------
# PERGUNTA
# -----------------------------------------------------------
pergunta = st.text_input("🔎 Pergunte algo sobre os PDFs carregados:")

if st.button("Enviar pergunta"):
    if st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado ainda.")
    elif not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        docs = st.session_state.vectorstore.similarity_search(pergunta, k=8)

        contexto = ""
        for d in docs:
            contexto += f"\n\n[PDF: {d.metadata.get('pdf_name')}] ---\n{d.page_content}"

        prompt = f"""
Responda SOMENTE com base no contexto abaixo.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])

        st.subheader("🧠 Resposta:")
        st.write(resposta.content)

        # TRECHOS USADOS
        st.markdown("---")
        st.subheader("📌 Trechos usados:")

        shown = set()
        for d in docs:
            trecho = d.page_content.strip()
            chave = trecho.replace("\n", " ")[:300]

            if chave in shown:
                continue
            shown.add(chave)

            st.write(f"📄 **{d.metadata.get('pdf_name')}**")
            st.write(trecho[:500] + "...")
