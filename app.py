import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG Multi-PDF + LangChain + Streamlit + GitHub 📚🚀")

api_key = st.secrets["OPENAI_API_KEY"]

# LLM (OpenAI) para gerar respostas
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini"
)

# Embeddings locais (sem limite, grátis)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Upload múltiplo de PDFs
uploaded_files = st.file_uploader(
    "Envie um ou vários PDFs para análise:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_documents = []

    for uploaded_file in uploaded_files:
        # Salvar temporário
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"{uploaded_file.name} carregado com sucesso!")

        # Carregar o PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Dividir em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(docs)

        # Adicionar ao conjunto geral
        all_documents.extend(docs)

    st.info("Todos os PDFs foram processados. Construindo o índice...")

    # Criar FAISS com todos os documentos juntos
    vectorstore = FAISS.from_documents(all_documents, embeddings)

    st.success("Todos os PDFs foram indexados! Faça sua pergunta:")

    pergunta = st.text_input("Faça uma pergunta sobre os PDFs:")

    if st.button("Enviar pergunta"):
        if pergunta:
            # Busca nos múltiplos PDFs
            docs = vectorstore.similarity_search(pergunta, k=4)

            # Construir contexto
            contexto = ""
            for d in docs:
                contexto += f"\n\n[PDF: {d.metadata.get('source', 'desconhecido')}] ---\n{d.page_content}"

            prompt = f"""
            Use o contexto abaixo para responder à pergunta.

            CONTEXTO:
            {contexto}

            PERGUNTA:
            {pergunta}
            """

            resposta = llm.invoke([HumanMessage(content=prompt)])
            st.write(resposta.content)
        else:
            st.warning("Por favor, digite uma pergunta.")
