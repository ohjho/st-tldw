import streamlit as st
import litellm, re, os
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()


def extract_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    # Handle different YouTube URL formats
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
        r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # If no pattern matches, assume the input is already a video ID
    if len(url) == 11 and re.match(r"^[a-zA-Z0-9_-]+$", url):
        return url

    return None


def get_youtube_transcript(video_id: str) -> dict:
    """Fetch transcript from YouTube video."""
    try:
        transcript_list = ytt_api.fetch(video_id)
        transcript_text = " ".join([item.text for item in transcript_list])
        return {
            "success": True,
            "transcript": transcript_text,
            "raw_transcript": transcript_list,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "transcript": None,
            "raw_transcript": None,
            "error": str(e),
        }


def youtube_transcript(model: str, temperature: float, max_tokens: int, api_key: str):
    """Extract and display YouTube transcript with optional AI summarization."""
    st.header("📺 YouTube Transcript Extractor")

    # Initialize session state for youtube
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "youtube_video_id" not in st.session_state:
        st.session_state.youtube_video_id = ""
    if "youtube_transcript" not in st.session_state:
        st.session_state.youtube_transcript = ""

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
                result = get_youtube_transcript(video_id)

                if result["success"]:
                    st.session_state.youtube_transcript = result["transcript"]
                    st.success("✅ Transcript extracted successfully!")
                else:
                    st.error(f"❌ Failed to fetch transcript: {result['error']}")

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
            st.text_area(
                "Full Transcript",
                value=st.session_state.youtube_transcript,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )
        with col2:
            st.metric(
                "Transcript Length",
                f"{len(st.session_state.youtube_transcript)} characters",
            )
            st.metric("Word Count", len(st.session_state.youtube_transcript.split()))

        # Copy to clipboard
        st.button("📋 Copy Transcript", key="copy_transcript")

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
            user_prompt = f"Please summarize this transcript:\n\n{st.session_state.youtube_transcript[:3000]}"
        elif analysis_type == "Ask Questions":
            user_input = st.text_input("Ask a question about the transcript content:")
            if user_input:
                system_prompt = "You are a helpful assistant. Answer questions about the provided YouTube transcript accurately and thoroughly."
                user_prompt = f"Transcript:\n{st.session_state.youtube_transcript[:3000]}\n\nQuestion: {user_input}"
            else:
                st.info("Enter a question to analyze the transcript.")
                return
        else:  # Extract Key Points
            system_prompt = "You are a helpful assistant. Extract and list the key points from the following YouTube transcript."
            user_prompt = f"Extract key points from this transcript:\n\n{st.session_state.youtube_transcript[:3000]}"

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

                    # Call litellm with streaming
                    stream = litellm.completion(
                        model=model,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        # stream=True,
                        api_key=api_key,
                    )

                    # Display streamed response
                    st.subheader("Analysis Result")
                    # response_content = st.write_stream(stream)
                    st.caption(f"response from {stream['model']}")
                    st.write(stream["choices"][0]["message"]["content"])

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


def chat_interface():
    """Main chat interface with litellm."""
    st.header("💬 Chat Interface")

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Chat Configuration")

        # Model selection
        model = st.selectbox(
            "Select Model",
            options=[
                "gpt-4",
                "gpt-4-turbo-preview",
                "gpt-3.5-turbo",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "mistral/mistral-7b-instruct",
                "ollama/neural-chat",
            ],
            help="Select an LLM model to use for chat",
            key="chat_model",
        )

        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your API key for the selected model provider",
            key="chat_api_key",
        )

        if api_key:
            st.session_state.api_key_set = True
            # Set the API key for litellm
            if "gpt" in model:
                litellm.api_key = api_key
            elif "claude" in model:
                litellm.api_key = api_key

        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more focused",
            key="chat_temperature",
        )

        # Max tokens slider
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=2048,
            step=100,
            help="Maximum length of the response",
            key="chat_max_tokens",
        )

        # System prompt
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful AI assistant.",
            height=100,
            help="Define the behavior and personality of the assistant",
            key="chat_system_prompt",
        )

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Display statistics
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Model", model.split("/")[-1][:15])

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if not st.session_state.api_key_set:
            st.error("Please enter an API key in the sidebar")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # Prepare messages for API call
                api_messages = [
                    {"role": "system", "content": system_prompt},
                    *[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                ]

                try:
                    # Call litellm with streaming
                    stream = litellm.completion(
                        model=model,
                        messages=api_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

                    # Use st.write_stream to display the streamed response
                    response_content = st.write_stream(stream)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    # Remove the user message if there was an error
                    st.session_state.messages.pop()
                    st.stop()

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )


def main():
    """Main application."""
    st.set_page_config(
        page_title="ST-TLDW",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 ST-TLDW: Streamlit YouTube Transcript & LLM Chat")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_key_set" not in st.session_state:
        st.session_state.api_key_set = False

    # Sidebar for analysis configuration
    with st.sidebar:
        st.title("⚙️ Analysis Configuration")

        # Model selection
        model = st.selectbox(
            "Select Model",
            options=[
                "openrouter/openrouter/free",
                "openrouter/google/gemma-3-4b-it:free",
                "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
            ],
            help="Select an LLM model for analysis",
        )

        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Enter your API key for the selected model provider",
            value=os.environ.get("OPENROUTER_API_KEY", ""),
        )

        if api_key:
            if "gpt" in model:
                litellm.api_key = api_key
            elif "claude" in model:
                litellm.api_key = api_key

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

    # Navigation tabs
    tab1, tab2 = st.tabs(["📺 YouTube Transcript", "💬 Chat"])

    with tab1:
        youtube_transcript(
            model=model, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )

    with tab2:
        st.write("coming soon...")
        # chat_interface()


if __name__ == "__main__":
    main()
