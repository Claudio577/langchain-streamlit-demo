if st.button("Enviar pergunta"):
    if st.session_state.vectorstore is None:
        st.error("Nenhum PDF carregado.")
    elif not pergunta:
        st.warning("Digite algo.")
    else:

        # Prefixo obrigatório do E5
        query = "query: " + pergunta

        # 1️⃣ Recuperar muitos trechos brutos do FAISS
        docs = st.session_state.vectorstore.similarity_search(query, k=12)

        # 2️⃣ RE-RANKING por LLM (🔥 deixa o RAG INSANO)
        docs = rerank_with_llm(llm, pergunta, docs)

        # 3️⃣ Selecionar apenas os melhores trechos re-rankeados
        top_docs = docs[:4]   # pegue só os 4 melhores

        # Construir contexto LIMPO
        contexto = ""
        for d in top_docs:
            texto = d.page_content.replace("passage: ", "")
            texto = clean_text_block(texto)
            contexto += f"\n\n[PDF: {d.metadata.get('pdf_name')}] ---\n{texto}"

        prompt = f"""
Use SOMENTE o contexto abaixo para responder da forma mais precisa possível.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:
"""

        resposta = llm.invoke([HumanMessage(content=prompt)])

        st.subheader("🧠 Resposta:")
        st.write(resposta.content)

        # Mostrar TRECHOS USADOS (só os rerankeados)
        st.markdown("---")
        st.subheader("📌 Trechos usados:")

        shown = set()
        for d in top_docs:
            texto = clean_text_block(d.page_content.replace("passage: ", ""))
            chave = texto.replace("\n", " ")[:300]

            if chave in shown:
                continue
            shown.add(chave)

            st.write(f"📄 **{d.metadata.get('pdf_name')}**")
            st.write(texto[:800] + "...")
