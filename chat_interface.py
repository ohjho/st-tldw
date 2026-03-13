import streamlit as st
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_srt(srt_string: str) -> list[Document]:
    """Split SRT string into document chunks, respecting subtitle block boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.create_documents([srt_string])


def _build_bm25_retriever(chunks: list[Document], k: int = 4) -> BM25Retriever:
    """Build a BM25 keyword retriever from document chunks."""
    retriever = BM25Retriever.from_documents(chunks, k=k)
    return retriever


def _build_faiss_retriever(chunks: list[Document], k: int = 4):
    """Build a FAISS semantic retriever using BGE embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": k})


def _stream_text(stream):
    """Yield text content from langchain AIMessageChunk stream."""
    for chunk in stream:
        if chunk.content:
            yield chunk.content


def chat_with_rag(
    srt_string: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    default_method: str = None,
):
    """RAG chat interface over an SRT transcript string."""
    st.header("💬 Chat with Transcript")

    if not srt_string:
        st.info(
            "Fetch a transcript in the YouTube tab first, then come here to chat about it."
        )
        return

    if not api_key:
        st.warning("API key required for RAG chat. Set it in the sidebar.")
        return

    # Initialize session state
    for key, default in [
        ("rag_messages", []),
        ("rag_retriever", None),
        ("rag_indexed_srt", ""),
        ("rag_retrieval_method", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Retrieval method selector
    method_options = ["BM25 (Keyword)", "Semantic (BGE)"]
    default_index = 1 if default_method and default_method.lower() == "semantic" else 0
    retrieval_method = st.radio(
        "Retrieval method",
        options=method_options,
        index=default_index,
        horizontal=True,
        help="BM25: fast keyword matching, no model download. Semantic: neural embeddings (~130MB first-run download), understands synonyms.",
    )

    # Build/rebuild retriever when transcript or method changes
    needs_rebuild = (
        srt_string != st.session_state.rag_indexed_srt
        or retrieval_method != st.session_state.rag_retrieval_method
    )

    if needs_rebuild:
        spinner_msg = (
            "Indexing transcript..."
            if "BM25" in retrieval_method
            else "Indexing transcript (downloading embedding model on first run)..."
        )
        with st.spinner(spinner_msg):
            chunks = _split_srt(srt_string)
            if "BM25" in retrieval_method:
                st.session_state.rag_retriever = _build_bm25_retriever(chunks)
            else:
                st.session_state.rag_retriever = _build_faiss_retriever(chunks)
            st.session_state.rag_indexed_srt = srt_string
            st.session_state.rag_retrieval_method = retrieval_method
            st.session_state.rag_messages = []

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.rag_messages = []
        st.rerun()

    # Render chat history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the transcript..."):
        # Display user message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant chunks
        retriever = st.session_state.rag_retriever
        relevant_docs = retriever.invoke(prompt)
        context = "\n\n".join(doc.page_content for doc in relevant_docs)

        # Build messages for LLM
        system_content = (
            "You are a helpful assistant answering questions about a YouTube video transcript. "
            "Use the following transcript excerpts to answer the user's question. "
            "If the answer is not in the provided context, say so.\n\n"
            f"Context:\n{context}"
        )
        messages = [SystemMessage(content=system_content)]
        for msg in st.session_state.rag_messages[
            :-1
        ]:  # exclude current user msg (added separately)
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))

        # Stream response
        llm = ChatLiteLLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            streaming=True,
        )

        with st.chat_message("assistant"):
            try:
                with st.spinner("thinking..."):
                    stream = llm.stream(messages)
                    response_content = st.write_stream(_stream_text(stream))
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.rag_messages.pop()
                return

        st.session_state.rag_messages.append(
            {"role": "assistant", "content": response_content}
        )
