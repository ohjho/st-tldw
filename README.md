# Too Long; Didn't Watch

A Streamlit app that extracts YouTube video transcripts (via SerpAPI) and lets you chat with them using RAG. 
Supports BM25 keyword retrieval or semantic search (BAAI/bge-small-en-v1.5 + FAISS). 
LLM calls go through litellm/OpenRouter.

For a detailed explanation of the architecture, see [the write-up](https://seekingvega.github.io/sv-journal/notebooks/writeup_tldw.html).

## Why?
The excellent [NotebookLM](https://notebooklm.google.com) can already do this better so why did I build this?

1. to learn Agentic Coding
2. to learn FastHTML: prototyped in Streamlit and asked Claude to teach me how to recreate this in [FastHTML](https://fastht.ml/docs/#getting-help-from-ai)
3. the created codebase could still be useful in cases where NotebookLM can't be used
