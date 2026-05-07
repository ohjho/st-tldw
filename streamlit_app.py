import os

import litellm
import streamlit as st
from dotenv import load_dotenv
from millify import prettify
from streamlit_float import float_init

from chat_interface import chat_with_rag
from utils import (
    copy_to_clipboard_button,
    detect_mobile_device,
    extract_video_id,
    get_ollama_free_models,
    get_serpapi_account_info,
    get_supadata_account_info,
    get_video_metadata_youtube_api,
    get_youtube_transcript_serpapi,
    get_youtube_transcript_supadata,
    hide_streamlit_chrome,
    raw_transcript_to_srt,
    render_markdown_with_timestamps,
)

if os.path.isfile("./secrets.env"):
    load_dotenv("./secrets.env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", None)
DEFAULT_OR_API_KEY = os.environ.get("OPENROUTER_API_KEY", None)
DEFAULT_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", None)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", None)
SUPADATA_API_KEY = os.environ.get("SUPADATA_API_KEY", None)

# Streamlit Cloud sets this env var automatically; use it to enable disk-backed
# cache persistence only in the deployed environment (not during local dev).
IS_STREAMLIT_CLOUD = os.environ.get("USER") == "appuser"
print(f"IS_STREAMLIT_CLOUD: {IS_STREAMLIT_CLOUD}")


@st.cache_data(show_spinner=False, persist=IS_STREAMLIT_CLOUD)
def _cached_analysis(
    video_id: str,
    analysis_type: str,
    _system_prompt: str,
    _user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    api_base: str = None,
    cache_version: int = 0,
) -> dict:
    """Run an LLM analysis call and cache the result.

    The cache key is based on ``video_id``, ``analysis_type``, and model
    parameters so re-running the same analysis is instant.  The full prompt
    strings are prefixed with ``_`` so Streamlit excludes them from the
    hash (they are derived deterministically from the other keys).

    Args:
        video_id: YouTube video ID (cache key).
        analysis_type: ``"Summarize"`` or ``"Extract Key Points"`` (cache key).
        _system_prompt: System prompt for the LLM (not hashed).
        _user_prompt: User prompt for the LLM (not hashed).
        model: LiteLLM model identifier.
        temperature: Sampling temperature.
        max_tokens: Max tokens for the response.
        api_key: Provider API key.
        api_base: Optional provider base URL.
        cache_version: Per-video invalidation counter. Bumping this for a
            given ``video_id`` forces a cache miss without clearing results
            cached for other videos.  Callers manage the counter in
            ``st.session_state.analysis_version``.

    Returns:
        dict with ``model`` and ``content`` keys.
    """
    api_messages = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": _user_prompt},
    ]
    completion_kwargs = dict(
        model=model,
        messages=api_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    if api_base:
        completion_kwargs["api_base"] = api_base
    response = litellm.completion(**completion_kwargs)
    return {
        "model": response["model"],
        "content": response["choices"][0]["message"]["content"],
    }


def analyze_transcript(
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    api_base: str = None,
):
    """Auto-run Summarize and Extract Key Points for a YouTube transcript.

    Reads the SRT transcript and video metadata from ``st.session_state``
    and runs both analyses automatically.  Results are cached per
    ``(video_id, analysis_type, model, temperature, max_tokens)``.

    Args:
        model: LiteLLM model identifier.
        temperature: Sampling temperature.
        max_tokens: Max tokens for the response.
        api_key: Provider API key.
        api_base: Optional provider base URL.
    """
    srt_string = ""
    raw = st.session_state.get("raw_transcript")
    if raw:
        srt_string = raw_transcript_to_srt(raw)

    if not srt_string:
        return

    if not api_key:
        st.warning("API key in the sidebar required for AI Analysis")
        return

    video_id = st.session_state.get("youtube_video_id", "")
    metadata = st.session_state.get("video_metadata") or {}
    title = metadata.get("title", "Unknown")
    description = metadata.get("description", "")

    col_label, col_btn = st.columns([90, 10])
    col_label.caption("AI Summary:")
    # Per-video cache invalidation: bump a version counter so only this
    # video's cached analysis is regenerated (other videos keep their cache).
    versions = st.session_state.setdefault("analysis_version", {})
    if col_btn.button(
        ":material/replay:",
        help="Clear cache and re-generate summary",
        type="tertiary",
    ):
        versions[video_id] = versions.get(video_id, 0) + 1
        st.rerun()

    context_block = (
        f"Video title: {title}\n"
        f"Video description: {description}\n\n"
        f"Transcript (SRT format with timestamps):\n{srt_string}"
    )

    system_prompt = (
        "You are a helpful assistant that analyzes YouTube video transcripts. "
        "The transcript is provided in SRT format with timestamps in "
        "HH:MM:SS,mmm format. Always cite the relevant timestamp "
        "(HH:MM:SS format) next to each point or claim so the reader "
        "can jump to that moment in the video."
    )

    analysis_prompt = (
        "Analyze the following YouTube video transcript. "
        "First, provide a concise summary highlighting the main points and key takeaways. "
        "Then, list the key points as a numbered list with the timestamp "
        "(HH:MM:SS) where each idea is discussed in the video.\n\n"
        f"{context_block}"
    )

    try:
        with st.spinner("Generating analysis..."):
            result = _cached_analysis(
                video_id=video_id,
                analysis_type="Summarize and Extract Key Points",
                _system_prompt=system_prompt,
                _user_prompt=analysis_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base,
                cache_version=versions.get(video_id, 0),
            )
        st.caption(f"response from {result['model']}")
        if video_id:
            render_markdown_with_timestamps(
                result["content"], video_id, open_in_new_tab=True, use_youtube_url=True
            )
        else:
            st.write(result["content"])
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")


def youtube_transcript(
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
    st_data_container,
    api_base: str = None,
):
    """Extract and display YouTube transcript with optional AI summarization."""
    # st.header("📺 YouTube Transcript Extractor")

    if not SERPAPI_KEY and not SUPADATA_API_KEY:
        st.error(
            "No transcript API key found. Set `SERPAPI_KEY` or `SUPADATA_API_KEY`."
        )
        return

    # Initialize session state for youtube
    stored_values = [
        "youtube_url",
        "youtube_video_id",
        "youtube_transcript",
        "raw_transcript",
        "video_metadata",
        "video_start_time",
        "transcript_source",
    ]
    for v in stored_values:
        if v not in st.session_state:
            st.session_state[v] = 0 if v == "video_start_time" else ""

    # URL input
    st.caption("too long; don't watch", help="open left sidebar for settings")
    col1, col2 = st.columns([90, 10])
    with col1:
        youtube_url = st.text_input(
            "Too Long; Don't Watch",
            placeholder="Enter YouTube URL, e.g. https://www.youtube.com/watch?v=... or paste video ID",
            help="open left sidebar for settings",
            label_visibility="collapsed",
        )
    with col2:
        fetch_button = st.button("GO", width="content")

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
                st.session_state.video_start_time = 0

                # Try SerpAPI first, fall back to Supadata
                result = None
                source = ""
                if SERPAPI_KEY:
                    result = get_youtube_transcript_serpapi(
                        video_id, serpapi_key=SERPAPI_KEY
                    )
                    if result["success"]:
                        source = "SerpAPI"

                if (result is None or not result["success"]) and SUPADATA_API_KEY:
                    result = get_youtube_transcript_supadata(
                        video_id, supadata_key=SUPADATA_API_KEY
                    )
                    if result["success"]:
                        source = "Supadata"

                if result and result["success"]:
                    st.session_state.youtube_transcript = result["transcript"]
                    st.session_state.raw_transcript = result["raw_transcript"]
                    st.session_state.transcript_source = source
                    st.session_state.video_metadata = get_video_metadata_youtube_api(
                        video_id, YOUTUBE_API_KEY
                    )
                    st.query_params["v"] = video_id
                    st.success("✅ Transcript extracted successfully!")
                else:
                    error_msg = (
                        result["error"] if result else "No transcript API configured"
                    )
                    st.error(
                        f"❌ Failed to fetch transcript for video id `{video_id}`: {error_msg}"
                    )
                    st.query_params.clear()
                    st.stop()

    # Display transcript if available
    if st.session_state.youtube_transcript:
        # Display video embed
        if st.session_state.youtube_video_id:
            # st.subheader("Video Preview")
            st.video(
                f"https://www.youtube.com/embed/{st.session_state.youtube_video_id}",
                start_time=st.session_state.get("video_start_time", 0),
                autoplay=st.session_state.video_start_time > 0,
                muted=True,
            )

        with st_data_container:
            # st.metric(
            #     "Transcript Length",
            #     f"{len(st.session_state.youtube_transcript)} characters",
            # )
            word_count = len(st.session_state.youtube_transcript.split())

            # Copy to clipboard and share link
            if st.session_state.youtube_video_id:
                base_url = os.getenv("APP_BASE_URL", "http://localhost:8501")
                share_url = f"{base_url}/?v={st.session_state.youtube_video_id}"
                st.caption("Share this link:")
                st.code(share_url, wrap_lines=True)
                # copy_to_clipboard_button(share_url)

            # Transcript display section
            tab_text, tab_json, tab_srt = st.tabs(
                [
                    ":material/article:",
                    ":material/data_object:",
                    ":material/closed_caption:",
                ]
            )
            transcript_source = st.session_state.get("transcript_source", "")
            with tab_text:
                st.text_area(
                    f"Transcript by `{transcript_source}` in plain text format ({prettify(word_count)} words):",
                    value=st.session_state.youtube_transcript,
                    height=300,
                    disabled=True,
                )

            # Show JSON/raw transcript returned by SerpApi
            with tab_json:
                st.caption(f"Raw `{transcript_source}` Transcript Data")
                st.write(st.session_state.raw_transcript)

            # Convert to SRT string and provide download button in the SRT tab
            raw = st.session_state.raw_transcript
            srt_string = ""
            if raw:
                try:
                    srt_string = raw_transcript_to_srt(raw)
                except Exception as e:
                    srt_string = ""
                    tab_srt.error(f"Failed to convert transcript to SRT: {e}")

            if srt_string:
                # Display SRT in a disabled text area for preview
                tab_srt.text_area(
                    "Transcript in SRT format",
                    value=srt_string,
                    height=300,
                    disabled=True,
                    help="the transcript in this format is consumed by the LLM",
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

        # AI Analysis section
        analyze_transcript(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
        )


def main():
    """Main application."""
    st.set_page_config(
        page_title="tl;dw",
        page_icon="asset/logo_icon.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.logo("asset/logo_large.png", size="large", icon_image="asset/logo_icon.png")
    # hide_streamlit_chrome()
    float_init()
    st.markdown(
        "<style>"
        ".block-container { padding-top: 1rem; }"
        " a[href='https://streamlit.io/cloud'][target='_blank'] { display: none; }"
        "</style>",
        unsafe_allow_html=True,
    )
    # st.title("🎬 ST-TLDW: Streamlit YouTube Transcript & LLM Chat")

    # Read URL query params for session sharing
    qp = st.query_params
    url_video_id = qp.get("v", None)
    url_method = qp.get("method", None)
    url_start_time = qp.get("t", None)
    if url_start_time is not None:
        try:
            st.session_state.video_start_time = int(url_start_time)
        except (ValueError, TypeError):
            st.session_state.video_start_time = 0

    # Initialize session state
    if not DEFAULT_OLLAMA_API_KEY:
        st.error(f"missing `DEFAULT_OLLAMA_API_KEY` in secrets. Please set it.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False

    # Auto-detect mobile for compact mode default
    if "compact_mode_user_set" not in st.session_state:
        st.session_state.compact_mode_user_set = False
    if not st.session_state.compact_mode_user_set:
        is_mobile = detect_mobile_device()
        if is_mobile is not None:
            st.session_state.compact_mode_value = is_mobile
            if is_mobile and "device_detected" not in st.session_state:
                st.session_state.device_detected = True
                st.rerun()
        else:
            st.session_state.setdefault("compact_mode_value", False)

    # Auto-load transcript from URL param ?v=VIDEO_ID
    if url_video_id and (SERPAPI_KEY or SUPADATA_API_KEY):
        if st.session_state.get("youtube_video_id") != url_video_id:
            result = None
            source = ""
            if SERPAPI_KEY:
                result = get_youtube_transcript_serpapi(
                    url_video_id, serpapi_key=SERPAPI_KEY
                )
                if result["success"]:
                    source = "SerpAPI"

            if (result is None or not result["success"]) and SUPADATA_API_KEY:
                result = get_youtube_transcript_supadata(
                    url_video_id, supadata_key=SUPADATA_API_KEY
                )
                if result["success"]:
                    source = "Supadata"

            if result and result["success"]:
                st.session_state.youtube_video_id = url_video_id
                st.session_state.youtube_transcript = result["transcript"]
                st.session_state.raw_transcript = result["raw_transcript"]
                st.session_state.transcript_source = source
                st.session_state.video_metadata = get_video_metadata_youtube_api(
                    url_video_id, YOUTUBE_API_KEY
                )
                st.session_state.youtube_url = (
                    f"https://www.youtube.com/watch?v={url_video_id}"
                )
            else:
                error_msg = (
                    result["error"] if result else "No transcript API configured"
                )
                st.warning(
                    f"Could not load transcript for video `{url_video_id}`: {error_msg}"
                )

    # Sidebar for analysis configuration
    with st.sidebar:
        cols = st.columns((1, 2))
        if cols[0].button(
            ":material/delete:",
            width="content",
            type="tertiary",
            help="clear context and start over!",
        ):
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

        compact_default = st.session_state.get("compact_mode_value", False)
        b_compact = cols[1].toggle(
            "compact mode",
            value=compact_default,
            help="best for mobile experience",
        )
        if b_compact != compact_default:
            st.session_state.compact_mode_user_set = True
        st.session_state.compact_mode_value = b_compact

        tab_settings, tab_metrics, tab_readme = st.tabs(
            [
                ":material/settings:",
                ":material/info:",
                ":material/article:",
            ]
        )
        with tab_readme:
            st.write(f"""
                Have you ever came across a video that's **too long** that you **didn't watch** (**TLDW**)
                on Youtube?

                if so then this streamlit app is for you!

                This app analyze a Youtube Video's transcript to provide:
                * AI Summary
                * RAG Chat interface

                Obviously it **works best on podcast type videos**.

                Read [this blog post](https://seekingvega.github.io/sv-journal/notebooks/writeup_tldw.html) on how this app was built.
                """)
            st.image("asset/qr.svg", caption="Scan for this App's URL")
        with tab_settings:
            st.caption(f"LLM settings:")
            # Provider selection
            provider = "Ollama"
            # provider = st.radio(
            #     "Provider",
            #     options=["Ollama Cloud", "OpenRouter"],
            #     horizontal=True,
            # )

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
                ollama_options = (
                    get_ollama_free_models(DEFAULT_OLLAMA_API_KEY)
                    if DEFAULT_OLLAMA_API_KEY
                    else ["gpt-oss:20b", "gpt-oss:120b", "glm-4.7"]
                )
                ollama_model = st.selectbox(
                    "Select Model",
                    options=ollama_options,
                    accept_new_options=True,
                    help="Select an [Ollama Cloud model](https://ollama.com/search?c=cloud) or type a custom name",
                )
                # ollama_chat/ routes to /api/chat (the Ollama Cloud chat
                # endpoint); the bare ollama/ prefix targets /api/generate
                model = f"ollama_chat/{ollama_model}"
                api_base = "https://ollama.com"

            # API Key input
            custom_api_key = None
            # custom_api_key = st.text_input(
            #     "API Key",
            #     type="password",
            #     help="Enter your own API key for the selected model provider",
            # )
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
                max_value=10000,
                value=4096,
                step=100,
            )

        with tab_metrics:
            cols = st.columns(2)
            if SERPAPI_KEY:
                serp_info = get_serpapi_account_info(SERPAPI_KEY)
                if serp_info:
                    cols[0].progress(
                        text=f"SerpAPI Usage ({serp_info['this_month_usage']}/{serp_info['searches_per_month']})",
                        value=serp_info["this_month_usage"]
                        / serp_info["searches_per_month"],
                    )
                else:
                    cols[0].warning("Could not fetch SerpAPI account info")
            if SUPADATA_API_KEY:
                supadata_info = get_supadata_account_info(SUPADATA_API_KEY)
                if supadata_info is not None:
                    cols[1].progress(
                        text=f"Supadata Usage ({supadata_info['usedCredits']}/{supadata_info['maxCredits']})",
                        value=supadata_info["usedCredits"]
                        / supadata_info["maxCredits"],
                    )
                else:
                    cols[1].warning(f"Could not fetch Supadata account info")

    # Navigation tabs
    # tab1, tab2 = st.tabs(["📺 YouTube Transcript", "💬 Chat"])
    lcol, rcol = (
        st.tabs([":material/wand_stars:", "Chat"]) if b_compact else st.columns((6, 4))
    )

    with lcol:
        youtube_transcript(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            st_data_container=tab_metrics,
        )

    with rcol:
        srt_for_rag = ""
        if st.session_state.get("raw_transcript"):
            srt_for_rag = raw_transcript_to_srt(st.session_state.raw_transcript)
        chat_with_rag(
            srt_string=srt_for_rag,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            default_method=url_method,
            api_base=api_base,
            st_settings_container=tab_settings,
        )


if __name__ == "__main__":
    main()
