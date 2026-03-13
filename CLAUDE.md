# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ST-TLDW (Streamlit YouTube Transcript & LLM Chat) is a Streamlit app that extracts YouTube video transcripts via SerpAPI, optionally analyzes them with LLMs through litellm/OpenRouter, and provides a RAG-powered chat interface to ask questions about the transcript content.

## Commands

```bash
# Run the app
uv run streamlit run streamlit_app.py

# Install dependencies
uv sync
```

## Architecture

### `streamlit_app.py` — Main app

- **`main()`** - Entry point. Sets up page config, sidebar controls (model, API key, temperature, max tokens), and two tabs: YouTube Transcript and Chat.
- **`youtube_transcript()`** - Core feature. Fetches transcripts via `get_youtube_transcript_serpapi()`, displays them in text/JSON/SRT tabs, and offers AI analysis (summarize, Q&A, key points) using litellm.
- **`get_youtube_transcript_serpapi()`** - Cached (`@st.cache_data(ttl="1d")`) SerpAPI call to fetch YouTube transcripts. Primary transcript source.
- **`get_youtube_transcript()`** - Alternative transcript fetcher using `youtube-transcript-api` directly (currently unused, kept as fallback).
- **`serp_transcript_to_srt()`** / **`ms_to_srt_timestamp()`** - Convert SerpAPI transcript format to SRT subtitle format with download support.
- **`chat_interface()`** - Legacy standalone chat UI (not wired up, superseded by RAG chat).

### `chat_interface.py` — RAG chat module

- **`chat_with_rag()`** - Main entry point. Renders the RAG chat UI inside the Chat tab. Accepts SRT string + LLM config from the sidebar. Users choose between two retrieval strategies via a radio toggle.
- **`_split_srt()`** - Splits SRT string into langchain `Document` chunks using `RecursiveCharacterTextSplitter` with `\n\n` and `\n` separators (respects subtitle block boundaries).
- **`_build_bm25_retriever()`** - BM25 keyword retrieval (instant, no model download). Uses `rank-bm25` via langchain's `BM25Retriever`.
- **`_build_faiss_retriever()`** - Semantic retrieval using `BAAI/bge-small-en-v1.5` embeddings + FAISS in-memory vector store (~130MB model download on first run).
- **`_stream_text()`** - Adapter generator that extracts `.content` from langchain `AIMessageChunk` objects for `st.write_stream`.

## Key Details

- **Python 3.10**, managed with `uv`
- **Environment**: API keys loaded from `secrets.env` (gitignored). Required: `SERPAPI_KEY`, `OPENROUTER_API_KEY`.
- **LLM routing**: Uses litellm with OpenRouter free-tier models by default (e.g., `openrouter/google/gemma-3-4b-it:free`). RAG chat uses langchain's `ChatLiteLLM` wrapper.
- **RAG dependencies**: langchain, langchain-community, faiss-cpu, rank-bm25, sentence-transformers, langchain-huggingface.
- **Session state keys**: `youtube_url`, `youtube_video_id`, `youtube_transcript`, `serp_transcript`, `messages`, `api_key_set`, `rag_messages`, `rag_retriever`, `rag_indexed_srt`, `rag_retrieval_method`.
