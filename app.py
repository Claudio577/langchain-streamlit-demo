import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================================
# FUNÇÃO 1 — REMOVER CABEÇALHOS DO DIÁRIO OFICIAL
# ==========================================================
def remove_governo_headers(text: str) -> str:
    """
    Remove cabeçalhos repetidos do Diário Oficial
    e assinaturas digitais que aparecem em TODAS as páginas.
    """
    linhas = text.split("\n")
    novas = []

    for linha in linhas:
        l = linha.strip()

        if "Este documento pode ser verificado pelo código" in l:
            continue
        if "https://www.doe.sp.gov.br/autenticidade" in l:
            continue
        if "Documento assinado digitalmente conforme" in l:
            continue
        if "ICP-Brasil" in l:
            continue
        if "/24" in l and "autenticidade" in l:
            continue

        novas.append(linha)

    return "\n".join(novas)


# ==========================================================
# FUNÇÃO 2 — NORMALIZAR TEXTO QUEBRADO EM VÁRIAS LINHAS
# ==========================================================
def clean_text_block(text: str) -> str:
    """
    Junta linhas quebradas (como PDFs ruins que quebram palavra por palavra)
    e monta frases legíveis.
    """
    lines = text.split("\n")
    new_lines = []
    buffer = ""

    for line in lines:
        line_strip = line.strip()

        if not line_strip:
            if buffer:
                new_lines.append(buffer)
                buffer = ""
            continue

        if len(line_strip.split()) <= 3:
            buffer += " " + line_strip
        else:
            if buffer and not buffer.endswith((".", "!", "?", ";", ":")):
                buffer += " " + line_strip
            else:
                if buffer:
                    new_lines.append(buffer)
                buffer = line_strip

    if buffer:
        new_lines.append(buffer)

    return "\n".join(new_lines)


# ==========================================================
# TÍTULO DO APP
# ==========================================================
st.title("📚 RAG Multi-PDF Inteligente – Sem Cabeçalhos Repetidos 🚀")


# ==========================================================
# LLM
# ==========================================================
api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)


# ==========================================================
# EMBEDDINGS
# ==========================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# ESTADOS DA SESSÃO
# ==========================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_list" not in st.session_state:
    st.session_state.pdf_list = []


# ==========================================================
# BOTÃO DE RESET TOTAL
# ==========================================================
st.markdown("### 🧹 Limpar PDFs carregados")

if st.button("🔄 Resetar memória e apagar todos os PDFs"):
    st.session_state.vectorstore = None
    st.session_state.pdf_list = []
    st.success("Memória limpa! Nenhum PDF carregado.")
    st.rerun()


# ==========================================================
# UPLOAD DE PDFs
# ==========================================================
uploaded_files = st.file_uploader(
    "Envie PDFs (um ou vários). Sempre será criado um índice novo:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

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

        # LIMPAR CABEÇALHOS ANTES DE INDEXAR
        for d in docs:
            texto = d.page_content
            texto = remove_governo_headers(texto)
            d.page_content = texto
            d.metadata["pdf_name"] = uploaded.name

        all_docs.extend(docs)
        st.session_state.pdf_list.append(uploaded.name)

    st.session_state.vectorstore = FAISS.from_documents(all_docs, embeddings)
    st.success("✨ Novo índice criado! PDFs limpos e prontos para perguntas.")


# ==========================================================
# PERGUNTA DO USUÁRIO
# ==========================================================
pergunta = st.text_input("🔎 Pergunte algo sobre os PDFs carregados:")

if st.button("Enviar pergunta"):
    if st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado ainda.")
    elif not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        docs = st.session_state.vectorstore.similarity_search(pergunta, k=8)

        # Montar contexto LIMPO
        contexto = ""
        for d in docs:
            texto_limpo = clean_text_block(d.page_content)
            contexto += f"\n\n[PDF: {d.metadata.get('pdf_name')}] ---\n{texto_limpo}"

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

        # ==========================================================
        # TRECHOS USADOS (LIMPINHOS E SEM DUPLICAÇÃO)
        # ==========================================================
        st.markdown("---")
        st.subheader("📌 Trechos usados:")

        shown = set()
        for d in docs:
            trecho = clean_text_block(d.page_content)
            chave = trecho.replace("\n", " ")[:300]

            if chave in shown:
                continue
            shown.add(chave)

            st.write(f"📄 **{d.metadata.get('pdf_name')}**")
            st.write(trecho[:800] + "...")
