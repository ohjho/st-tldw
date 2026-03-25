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

# Run tests
uv run pytest
```

## Architecture

### `streamlit_app.py` — Main app

- **`main()`** - Entry point. Sets up page config, sidebar controls (model, API key, temperature, max tokens), and two tabs: YouTube Transcript and Chat.
- **`youtube_transcript()`** - Core feature. Fetches transcripts via `get_youtube_transcript_serpapi()`, displays them in text/JSON/SRT tabs, and delegates AI analysis to `analyze_transcript()`.
- **`analyze_transcript()`** - Renders the AI analysis UI (Summarize / Extract Key Points). Reads SRT transcript and video metadata from session state, builds timestamp-aware prompts with video title and description, and calls the cached `_cached_analysis()` helper.
- **`_cached_analysis()`** - `@st.cache_data`-decorated LLM call via litellm. Cache key is `(video_id, analysis_type, model, temperature, max_tokens)` to avoid hashing large transcript strings.

### `utils.py` — Utility functions

- **`extract_video_id()`** - Extract video ID from various YouTube URL formats (standard, short, embed, bare ID).
- **`get_youtube_transcript_serpapi()`** - Cached (`@st.cache_data(ttl="1d")`) SerpAPI call to fetch YouTube transcripts. Primary transcript source.
- **`get_youtube_transcript()`** - Alternative transcript fetcher using `youtube-transcript-api` directly (currently unused, kept as fallback).
- **`get_video_metadata_oembed()`** - Cached (`@st.cache_data(ttl="1d")`) call to the free YouTube oEmbed endpoint. Returns video title, author name/URL, and thumbnail URL. No API key required.
- **`get_video_metadata_youtube_api()`** - Cached (`@st.cache_data(ttl="1d")`) call to the YouTube Data API v3. Returns rich metadata: title, description, channel, publish date, tags, view/like counts, duration, and thumbnail. Requires `YOUTUBE_API_KEY`.
- **`get_serpapi_searches_left()`** - Cached (5min TTL) call to SerpAPI account endpoint; displays remaining searches in the sidebar.
- **`serp_transcript_to_srt()`** / **`ms_to_srt_timestamp()`** - Convert SerpAPI transcript format to SRT subtitle format with download support.
- **`copy_to_clipboard_button()`** - Render an HTML button via `st.html` that copies arbitrary text to the clipboard using `navigator.clipboard`.
- **`render_markdown_with_timestamps()`** - Render LLM markdown with clickable timestamp links. Regex-finds `H:MM:SS`/`HH:MM:SS` patterns and converts them to `<a>` links via `st.markdown(unsafe_allow_html=True)`. Supports `open_in_new_tab` (adds `target="_blank"`) and `use_youtube_url` (links to youtube.com instead of in-app relative URL). Reuses `srt_timestamp_to_seconds()` for conversion.
- **`srt_timestamp_to_seconds()`** - Convert SRT timestamp (`HH:MM:SS,mmm`) to whole seconds for YouTube `&t=` URL parameters. Also handles `H:MM:SS` without milliseconds.
- **`detect_mobile_device()`** - Uses `streamlit-js-eval` to query `window.innerWidth` from the browser. Returns `True` (mobile, ≤600px), `False` (desktop), or `None` (JS not yet responded). Used to auto-default compact mode on mobile devices.

### `chat_interface.py` — RAG chat module

- **`chat_with_rag()`** - Main entry point. Renders the RAG chat UI inside the Chat tab. Accepts SRT string + LLM config from the sidebar. Users choose between two retrieval strategies via a radio toggle.
- **`_split_srt()`** - Splits SRT string into langchain `Document` chunks using `RecursiveCharacterTextSplitter` with `\n\n` and `\n` separators (respects subtitle block boundaries).
- **`_build_bm25_retriever()`** - BM25 keyword retrieval (instant, no model download). Uses `rank-bm25` via langchain's `BM25Retriever`.
- **`_build_faiss_retriever()`** - Semantic retrieval using `BAAI/bge-small-en-v1.5` embeddings + FAISS in-memory vector store (~130MB model download on first run).
- **`_stream_text()`** - Adapter generator that extracts `.content` from langchain `AIMessageChunk` objects for `st.write_stream`.

## Key Details

- **Python 3.10**, managed with `uv`
- **Environment**: API keys loaded from `secrets.env` (gitignored). Required: `SERPAPI_KEY`, `OPENROUTER_API_KEY`. Optional: `YOUTUBE_API_KEY` (for YouTube Data API v3 metadata).
- **LLM routing**: Uses litellm with OpenRouter free-tier models by default (e.g., `openrouter/google/gemma-3-4b-it:free`). RAG chat uses langchain's `ChatLiteLLM` wrapper.
- **RAG dependencies**: langchain, langchain-community, faiss-cpu, rank-bm25, sentence-transformers, langchain-huggingface.
- **Device detection**: `streamlit-js-eval` queries `window.innerWidth` to auto-enable compact mode on mobile (≤600px). Detection runs once on first load; user can override via the sidebar toggle.
- **URL query params**: `?v=VIDEO_ID` auto-loads a transcript on page load; `?method=semantic` pre-selects the RAG retrieval method; `?t=SECONDS` seeks the video to a specific timestamp (used by clickable timestamp links).
- **Session state keys**: `youtube_url`, `youtube_video_id`, `youtube_transcript`, `serp_transcript`, `video_metadata`, `video_start_time`, `messages`, `api_key_set`, `rag_messages`, `rag_retriever`, `rag_indexed_srt`, `rag_retrieval_method`, `compact_mode_user_set`, `compact_mode_value`, `device_detected`.

## Tests

Tests live in the `tests/` directory. Pytest is configured in `pyproject.toml` with `pythonpath = ["."]` so test files can import root-level modules directly (e.g., `from utils import ...`).

- `tests/test_utils.py` — unit tests for pure utility functions (no network)
- `tests/test_oembed.py` — integration tests for oEmbed metadata fetcher (hits network)
- `tests/test_youtube_api.py` — integration tests for YouTube Data API (requires `YOUTUBE_API_KEY`)

## Guidelines

- Create tests in `tests/` and update CLAUDE.md for each new feature
- use Google-style docstring for new functions and add a doctest compatible unit test if possible
- keep code modular to ensure ease in future refactoring
- Prefer native Streamlit features over custom CSS
- Keep custom CSS minimal
