import os

import litellm
import streamlit as st
from dotenv import load_dotenv

from chat_interface import chat_with_rag
from utils import (
    extract_video_id,
    get_serpapi_searches_left,
    get_video_metadata_youtube_api,
    get_youtube_transcript_serpapi,
    serp_transcript_to_srt,
)

if os.path.isfile("./secrets.env"):
    load_dotenv("./secrets.env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", None)
DEFAULT_OR_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
DEFAULT_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", None)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", None)


def youtube_transcript(
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    api_base: str = None,
):
    """Extract and display YouTube transcript with optional AI summarization."""
    st.header("📺 YouTube Transcript Extractor")

    assert SERPAPI_KEY, f"SERPAPI_KEY not found in ENV"

    # Initialize session state for youtube
    stored_values = [
        "youtube_url",
        "youtube_video_id",
        "youtube_transcript",
        "serp_transcript",
        "video_metadata",
    ]
    for v in stored_values:
        if v not in st.session_state:
            st.session_state[v] = ""

    # URL input
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        youtube_url = st.text_input(
            "Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=... or paste video ID",
            help="Paste a YouTube URL or just the video ID",
        )
    with col2:
        fetch_button = st.button("Fetch Transcript", use_container_width=True)

    # Fetch transcript if button clicked
    if fetch_button and youtube_url:
        with st.spinner("Extracting transcript..."):
            video_id = extract_video_id(youtube_url)

            if not video_id:
                st.error(
                    "❌ Invalid YouTube URL or video ID. Please check and try again."
                )
            else:
                st.session_state.youtube_video_id = video_id
                # result = get_youtube_transcript(video_id)
                result = get_youtube_transcript_serpapi(
                    video_id, serpapi_key=SERPAPI_KEY
                )

                if result["success"]:
                    st.session_state.youtube_transcript = result["transcript"]
                    st.session_state.serp_transcript = result["raw_transcript"]
                    st.session_state.video_metadata = get_video_metadata_youtube_api(
                        video_id, YOUTUBE_API_KEY
                    )
                    st.query_params["v"] = video_id
                    st.success("✅ Transcript extracted successfully!")
                else:
                    st.error(
                        f"❌ Failed to fetch transcript for video id `{video_id}`: {result['error']}"
                    )

    # Display transcript if available
    if st.session_state.youtube_transcript:
        st.divider()

        # Display video embed
        if st.session_state.youtube_video_id:
            st.subheader("Video Preview")
            st.video(
                f"https://www.youtube.com/embed/{st.session_state.youtube_video_id}"
            )

        # Transcript display section
        st.subheader("📄 Transcript")

        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            tab_text, tab_json, tab_srt = st.tabs(
                [
                    ":material/article:",
                    ":material/data_object:",
                    ":material/closed_caption:",
                ]
            )
            tab_text.text_area(
                "Full Transcript",
                value=st.session_state.youtube_transcript,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

            # Show JSON/raw transcript returned by SerpApi
            tab_json.write(st.session_state.serp_transcript)

            # Convert to SRT string and provide download button in the SRT tab
            serp = st.session_state.serp_transcript
            srt_string = ""
            if serp:
                try:
                    srt_string = serp_transcript_to_srt(serp)
                except Exception as e:
                    srt_string = ""
                    tab_srt.error(f"Failed to convert transcript to SRT: {e}")

            if srt_string:
                # Display SRT in a disabled text area for preview
                tab_srt.text_area(
                    "SRT Preview",
                    value=srt_string,
                    height=300,
                    disabled=True,
                    label_visibility="collapsed",
                )
                # Provide download button
                file_name = (
                    f"{st.session_state.youtube_video_id}.srt"
                    if st.session_state.youtube_video_id
                    else "transcript.srt"
                )
                # st.download_button can accept string content directly
                tab_srt.download_button(
                    label="⬇️ Download .srt",
                    data=srt_string,
                    file_name=file_name,
                    mime="text/srt",
                )
            else:
                tab_srt.info("No timed transcript available to convert to SRT.")

        with col2:
            st.metric(
                "Transcript Length",
                f"{len(st.session_state.youtube_transcript)} characters",
            )
            st.metric("Word Count", len(st.session_state.youtube_transcript.split()))

        # Copy to clipboard and share link
        col_copy, col_share = st.columns(2)
        with col_copy:
            st.button("📋 Copy Transcript", key="copy_transcript")
        with col_share:
            if st.session_state.youtube_video_id:
                base_url = os.getenv("APP_BASE_URL", "http://localhost:8501")
                share_url = f"{base_url}/?v={st.session_state.youtube_video_id}"
                st.code(share_url, language=None)

        st.divider()

        # AI Analysis section
        st.subheader("🤖 AI Analysis")

        analysis_type = st.radio(
            "Select analysis type",
            options=["Summarize", "Ask Questions", "Extract Key Points"],
            horizontal=True,
        )

        # Analysis prompts
        if analysis_type == "Summarize":
            system_prompt = "You are a helpful assistant. Summarize the following YouTube transcript concisely, highlighting the main points and key takeaways."
            user_prompt = f"Please summarize this transcript:\n\n{st.session_state.youtube_transcript}"
        elif analysis_type == "Ask Questions":
            user_input = st.text_input("Ask a question about the transcript content:")
            if user_input:
                system_prompt = "You are a helpful assistant. Answer questions about the provided YouTube transcript accurately and thoroughly."
                user_prompt = f"Transcript:\n{st.session_state.youtube_transcript}\n\nQuestion: {user_input}"
            else:
                st.info("Enter a question to analyze the transcript.")
                return
        else:  # Extract Key Points
            system_prompt = "You are a helpful assistant. Extract and list the key points from the following YouTube transcript."
            user_prompt = f"Extract key points from this transcript:\n\n{st.session_state.youtube_transcript}"

        # Analyze button
        if not api_key:
            st.warning("API key in the sidebar required for AI Analysis")
            return None

        if st.button("🔍 Analyze", use_container_width=True):
            with st.spinner("Analyzing transcript..."):
                try:
                    api_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    # Call litellm
                    completion_kwargs = dict(
                        model=model,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                    )
                    if api_base:
                        completion_kwargs["api_base"] = api_base
                    stream = litellm.completion(**completion_kwargs)

                    # Display streamed response
                    st.subheader("Analysis Result")
                    st.caption(f"response from {stream['model']}")
                    st.write(stream["choices"][0]["message"]["content"])

                    # for unsupported stream, we need to write a wrapper: https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream
                    # response_content = st.write_stream(stream)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


def main():
    """Main application."""
    st.set_page_config(
        page_title="ST-TLDW",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 ST-TLDW: Streamlit YouTube Transcript & LLM Chat")

    # Read URL query params for session sharing
    qp = st.query_params
    url_video_id = qp.get("v", None)
    url_method = qp.get("method", None)

    # Initialize session state
    if not DEFAULT_OLLAMA_API_KEY:
        st.error(f"missing `DEFAULT_OLLAMA_API_KEY` in secrets. Please set it.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False

    # Auto-load transcript from URL param ?v=VIDEO_ID
    if url_video_id and SERPAPI_KEY:
        if st.session_state.get("youtube_video_id") != url_video_id:
            result = get_youtube_transcript_serpapi(
                url_video_id, serpapi_key=SERPAPI_KEY
            )
            if result["success"]:
                st.session_state.youtube_video_id = url_video_id
                st.session_state.youtube_transcript = result["transcript"]
                st.session_state.serp_transcript = result["raw_transcript"]
                st.session_state.video_metadata = get_video_metadata_youtube_api(
                    url_video_id, YOUTUBE_API_KEY
                )
                st.session_state.youtube_url = (
                    f"https://www.youtube.com/watch?v={url_video_id}"
                )
            else:
                st.warning(
                    f"Could not load transcript for video `{url_video_id}`: {result['error']}"
                )

    # Sidebar for analysis configuration
    with st.sidebar:
        st.title("⚙️ Analysis Configuration")

        # Provider selection
        provider = st.radio(
            "Provider",
            options=["Ollama Cloud", "OpenRouter"],
            horizontal=True,
        )

        api_base = None

        if provider == "OpenRouter":
            model = st.selectbox(
                "Select Model",
                options=[
                    "openrouter/openrouter/free",
                    "openrouter/google/gemma-3-4b-it:free",
                    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
                ],
                help="Select an OpenRouter model for analysis",
            )
        else:
            ollama_model = st.selectbox(
                "Select Model",
                options=[
                    "qwen3.5",
                    "deepseek-v3.2",
                    "gemini-3-flash-preview",
                ],
                accept_new_options=True,
                help="Select an Ollama Cloud model or type a custom name",
            )
            model = f"ollama/{ollama_model}"
            api_base = "https://ollama.com"

        # API Key input
        custom_api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your own API key for the selected model provider",
        )
        api_key = custom_api_key if custom_api_key else DEFAULT_OLLAMA_API_KEY

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=2048,
            step=100,
        )

        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

        if SERPAPI_KEY:
            searches_left = get_serpapi_searches_left(SERPAPI_KEY)
            if searches_left is not None:
                st.metric("SerpAPI Searches Left", searches_left)
            else:
                st.warning("Could not fetch SerpAPI account info")

    # Navigation tabs
    tab1, tab2 = st.tabs(["📺 YouTube Transcript", "💬 Chat"])

    with tab1:
        youtube_transcript(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
        )

    with tab2:
        srt_for_rag = ""
        if st.session_state.get("serp_transcript"):
            srt_for_rag = serp_transcript_to_srt(st.session_state.serp_transcript)
        chat_with_rag(
            srt_string=srt_for_rag,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            default_method=url_method,
            api_base=api_base,
        )


if __name__ == "__main__":
    main()
