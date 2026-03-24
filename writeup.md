---
title: "TLDW: Too Long; Didn't Watch"
subtitle: "Building a YouTube Transcript Analyzer with Streamlit, LLMs, and RAG"
format:
  html:
    code-fold: true
    toc: true
    toc-depth: 3
---

## Introduction

**TLDW** (Too Long; Didn't Watch) is a Streamlit application that extracts YouTube video transcripts, generates AI-powered summaries with clickable timestamps, and provides a RAG-based chat interface for asking questions about the video content.

The app is built around three core capabilities:

1. **Transcript retrieval** — fetching timestamped transcripts from YouTube via SerpAPI
2. **AI analysis** — generating summaries and key points using LLMs through LiteLLM
3. **RAG chat** — a retrieval-augmented chat interface for conversational Q&A over the transcript

This post walks through how each of these pieces works under the hood.

## Transcript Retrieval

The first step in the pipeline is obtaining a timestamped transcript for a given YouTube video. TLDW uses SerpAPI's YouTube transcript engine as its primary source, with video metadata enriched via the YouTube Data API.

### How It Works

When a user enters a YouTube URL (or the app loads one from the `?v=` query parameter), the app:

1. Extracts the video ID from the URL using regex patterns that handle standard, short, and embed URL formats
2. Calls SerpAPI's `youtube_video_transcript` engine to fetch the raw transcript
3. Fetches video metadata (title, description, tags, view counts) via the YouTube Data API v3
4. Converts the raw transcript into SRT subtitle format for downstream use

```{mermaid}
flowchart TD
    A[User enters YouTube URL] --> B[extract_video_id]
    B --> C{Valid video ID?}
    C -->|No| D[Show error]
    C -->|Yes| E[get_youtube_transcript_serpapi]
    E --> F{Success?}
    F -->|No| D
    F -->|Yes| G[Store raw transcript in session state]
    G --> H[get_video_metadata_youtube_api]
    H --> I[Store metadata in session state]
    I --> J[serp_transcript_to_srt]
    J --> K[SRT-formatted transcript ready]
    K --> L[Display transcript + video player]
```

### SerpAPI Transcript Format

SerpAPI returns the transcript as a list of segments, each with millisecond-precision timing:

```json
[
  {
    "start_ms": 0,
    "end_ms": 5040,
    "snippet": "Welcome to the video..."
  },
  {
    "start_ms": 5040,
    "end_ms": 10200,
    "snippet": "Today we're going to talk about..."
  }
]
```

The `get_youtube_transcript_serpapi()` function is cached with a one-day TTL using `@st.cache_data(ttl="1d")`, so repeated requests for the same video hit the cache instead of consuming API quota.

### SRT Conversion

The raw transcript is converted to SRT format by `serp_transcript_to_srt()`, which maps each segment's millisecond timestamps to the standard `HH:MM:SS,mmm` format:

```
1
00:00:00,000 --> 00:00:05,040
Welcome to the video...

2
00:00:05,040 --> 00:00:10,200
Today we're going to talk about...
```

SRT format is used throughout the app because it preserves timing information in a human-readable way that LLMs can parse and reference in their output.

### Video Metadata

TLDW enriches the transcript with video metadata from the YouTube Data API v3 (`get_video_metadata_youtube_api()`). This provides the video title, description, channel name, publish date, tags, and view/like counts. The title and description are passed into the AI analysis prompt so the LLM has full context about the video's topic.

For environments without a YouTube API key, a lightweight fallback uses the free oEmbed endpoint (`get_video_metadata_oembed()`) which returns basic info (title, author, thumbnail) without authentication.

### URL-Based Auto-Loading

The app supports deep linking via query parameters. A URL like `?v=dQw4w9WgXcQ` will automatically load the transcript on page load, and `?t=90` will seek the embedded video player to the 1:30 mark. This enables shareable links and powers the clickable timestamp feature in AI analysis output.


## AI Analysis

Once a transcript is loaded, TLDW can generate an AI-powered analysis that combines a concise summary with a numbered list of key points — each annotated with clickable timestamps that link back to the relevant moment in the video.

### Analysis Pipeline

```{mermaid}
flowchart TD
    A[SRT transcript in session state] --> B[Build context block]
    C[Video metadata: title + description] --> B
    B --> D[Construct system prompt]
    B --> E[Construct analysis prompt]
    D --> F[LiteLLM completion call]
    E --> F
    F --> G[LLM generates markdown with timestamps]
    G --> H[render_markdown_with_timestamps]
    H --> I[Regex finds HH:MM:SS patterns]
    I --> J[Convert to clickable YouTube links]
    J --> K[Render HTML with st.markdown]
```

### Prompt Construction

The analysis uses a two-message prompt structure — a system message that establishes the role and timestamp citation requirements, and a user message containing the full context:

**System prompt:**

> You are a helpful assistant that analyzes YouTube video transcripts. The transcript is provided in SRT format with timestamps in HH:MM:SS,mmm format. Always cite the relevant timestamp (HH:MM:SS format) next to each point or claim so the reader can jump to that moment in the video.

**User prompt:**

> Analyze the following YouTube video transcript. First, provide a concise summary highlighting the main points and key takeaways. Then, list the key points as a numbered list with the timestamp (HH:MM:SS) where each idea is discussed in the video.
>
> Video title: {title}
> Video description: {description}
>
> Transcript (SRT format with timestamps):
> {srt_string}

By explicitly instructing the LLM to cite timestamps in `HH:MM:SS` format, the output becomes machine-parseable for the next step — turning those timestamps into clickable links.

### Caching Strategy

The `_cached_analysis()` function wraps the LiteLLM call with `@st.cache_data`. The cache key is `(video_id, analysis_type, model, temperature, max_tokens)` — notably excluding the actual prompt strings (which are passed as underscore-prefixed parameters that Streamlit's caching ignores). This avoids hashing large transcript strings while still invalidating the cache when the user changes model settings.

### Clickable Timestamps

After the LLM returns its markdown response, `render_markdown_with_timestamps()` post-processes the output:

1. A regex `\b(\d{1,2}:\d{2}:\d{2})\b` finds all timestamp patterns like `1:23:45` or `00:05:30`
2. Each match is converted to seconds via `srt_timestamp_to_seconds()`
3. The timestamp text is wrapped in an `<a>` tag linking to `https://www.youtube.com/watch?v={video_id}&t={seconds}`
4. The result is rendered with `st.markdown(unsafe_allow_html=True)`

The links open in a new tab, so the user can jump to any cited moment in the video without leaving the app.


## RAG Chat Interface

Beyond one-shot analysis, TLDW provides a conversational chat interface powered by Retrieval-Augmented Generation (RAG). Users can ask follow-up questions about the video, and the system retrieves relevant transcript chunks to ground the LLM's responses.

### RAG Architecture

```{mermaid}
flowchart TD
    A[SRT transcript] --> B[_split_srt]
    B --> C[Document chunks]
    C --> D{Retrieval method?}
    D -->|BM25 Keyword| E[_build_bm25_retriever]
    D -->|Semantic| F[_build_faiss_retriever]
    E --> G[BM25Retriever]
    F --> H[FAISS vector store]
    H --> G2[FAISS Retriever]

    I[User question] --> J[retriever.invoke]
    G --> J
    G2 --> J
    J --> K[Top-k relevant chunks]
    K --> L[Build system prompt with context]
    M[Chat history] --> N[Build message stack]
    L --> N
    I --> N
    N --> O[ChatLiteLLM.stream]
    O --> P[Stream response to UI]
```

### Chunking the Transcript

The `_split_srt()` function splits the SRT transcript into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`:

- **Chunk size:** 1,000 characters
- **Overlap:** 200 characters
- **Separators:** `["\n\n", "\n"]` — prioritizes splitting on double newlines (SRT block boundaries) before falling back to single newlines

The overlap ensures that context isn't lost at chunk boundaries — if a topic spans two chunks, the overlapping region preserves continuity.

### Two Retrieval Strategies

Users choose between two retrieval methods via a radio toggle:

#### BM25 (Keyword)

`_build_bm25_retriever()` creates a `BM25Retriever` from LangChain. BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency and inverse document frequency. It's fast (no model download required) and effective for queries that use the same terminology as the transcript.

#### Semantic (FAISS + BGE)

`_build_faiss_retriever()` builds a vector store using:

- **Embeddings:** `BAAI/bge-small-en-v1.5` — a compact 384-dimensional embedding model (~130MB download on first run)
- **Vector store:** FAISS (Facebook AI Similarity Search) — an in-memory approximate nearest neighbor index

Semantic retrieval understands meaning rather than just keywords, so it can match questions about "revenue growth" to transcript segments discussing "increased sales by 40%."

Both retrievers return a configurable number of chunks (default 5, adjustable from 1–10 via a slider).

### Context Injection

When the user sends a message, the retriever fetches the top-k most relevant chunks. These are joined and injected into the system prompt:

> You are a helpful assistant answering questions about a YouTube video transcript. Use the following transcript excerpts to answer the user's question. If the answer is not in the provided context, say so.
>
> Context:
> {retrieved chunks}

The full message stack sent to the LLM includes:

1. The system message (with retrieved context)
2. The conversation history (previous user/assistant turns)
3. The current user question

This preserves multi-turn context while grounding each response in the relevant transcript excerpts.

### Streaming

The chat uses LangChain's `ChatLiteLLM` wrapper with `streaming=True`. The `_stream_text()` generator extracts `.content` from each `AIMessageChunk`, and `st.write_stream()` renders tokens to the UI in real-time as they arrive.

```{mermaid}
sequenceDiagram
    participant User
    participant Streamlit
    participant Retriever
    participant LLM

    User->>Streamlit: Ask question
    Streamlit->>Retriever: retriever.invoke(question)
    Retriever-->>Streamlit: Top-k transcript chunks
    Streamlit->>LLM: System prompt + context + chat history + question
    loop Token streaming
        LLM-->>Streamlit: AIMessageChunk
        Streamlit-->>User: Render token
    end
```

### Session State & Rebuilding

The retriever is stored in `st.session_state.rag_retriever` and only rebuilt when the transcript, retrieval method, or chunk count changes. This avoids re-indexing on every message. When the retriever is rebuilt, the chat history is cleared since the context basis has changed.


## Conclusion

TLDW chains together three layers — transcript extraction, LLM analysis, and retrieval-augmented chat — to make long-form video content quickly navigable. SerpAPI provides the raw transcript data, LiteLLM abstracts away model routing, and LangChain handles the RAG plumbing. Streamlit ties it all together with caching, session state, and streaming to keep the interface responsive.

The clickable timestamp feature is a small detail that makes a big difference: by instructing the LLM to cite timestamps and then post-processing its output with regex, the AI summary becomes a direct navigation tool for the video itself.
