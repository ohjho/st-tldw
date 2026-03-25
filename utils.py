"""Utility functions for ST-TLDW: transcript fetching, parsing, and metadata."""

import re
from typing import Dict, List, Optional

import httpx
import requests
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from various YouTube URL formats.

    Args:
        url: A YouTube URL or bare video ID.

    Returns:
        The 11-character video ID, or ``None`` if extraction fails.

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("not-a-url") is None
        True
    """
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


@st.cache_data(ttl=300)
def get_serpapi_searches_left(api_key: str) -> Optional[int]:
    """Fetch total searches left from SerpAPI account endpoint.

    Args:
        api_key: SerpAPI API key.

    Returns:
        Number of searches remaining, or ``None`` on failure.
    """
    try:
        resp = requests.get(
            "https://serpapi.com/account.json",
            params={"api_key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("total_searches_left")
    except Exception:
        return None


def get_youtube_transcript(video_id: str) -> dict:
    """Fetch transcript from YouTube video using youtube-transcript-api.

    Args:
        video_id: YouTube video ID.

    Returns:
        A dict with keys ``success``, ``transcript``, ``raw_transcript``, ``error``.
    """
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


@st.cache_data(ttl="1d")
def get_youtube_transcript_serpapi(video_id: str, serpapi_key: str) -> dict:
    """Fetch transcript from YouTube video using SerpApi.

    Args:
        video_id: YouTube video ID.
        serpapi_key: SerpAPI API key.

    Returns:
        A dict with keys ``success``, ``transcript``, ``raw_transcript``, ``error``.
    """
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "youtube_video_transcript",
            "v": video_id,
            "api_key": serpapi_key,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "transcript" in data:
            transcript_text = " ".join([item["snippet"] for item in data["transcript"]])
            return {
                "success": True,
                "transcript": transcript_text,
                "raw_transcript": data["transcript"],
                "error": None,
            }
        else:
            return {
                "success": False,
                "transcript": None,
                "raw_transcript": None,
                "error": data.get("error", "Transcript not found"),
            }
    except Exception as e:
        return {
            "success": False,
            "transcript": None,
            "raw_transcript": None,
            "error": str(e),
        }


@st.cache_data(ttl="1d")
def get_video_metadata_oembed(video_id: str) -> dict:
    """Fetch video metadata from the YouTube oEmbed endpoint.

    Uses the free, unauthenticated YouTube oEmbed API to retrieve basic
    metadata such as title, author name, author URL, and thumbnail URL.

    Args:
        video_id: YouTube video ID (e.g. ``"dQw4w9WgXcQ"``).

    Returns:
        A dict with keys ``success``, ``title``, ``author_name``,
        ``author_url``, ``thumbnail_url``, and ``error``.

    Examples:
        >>> result = get_video_metadata_oembed.__wrapped__("dQw4w9WgXcQ")
        >>> result["success"]
        True
        >>> result["title"] != ""
        True
    """
    url = (
        f"https://www.youtube.com/oembed"
        f"?url=https://www.youtube.com/watch?v={video_id}&format=json"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "success": True,
            "title": data.get("title", ""),
            "author_name": data.get("author_name", ""),
            "author_url": data.get("author_url", ""),
            "thumbnail_url": data.get("thumbnail_url", ""),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "title": "",
            "author_name": "",
            "author_url": "",
            "thumbnail_url": "",
            "error": str(e),
        }


@st.cache_data(ttl="1d")
def get_video_metadata_youtube_api(video_id: str, api_key: str) -> dict:
    """Fetch video metadata from the YouTube Data API v3.

    Retrieves rich metadata including description, tags, view/like counts,
    and duration using an authenticated YouTube Data API request.

    Args:
        video_id: YouTube video ID (e.g. ``"dQw4w9WgXcQ"``).
        api_key: YouTube Data API v3 key.

    Returns:
        A dict with keys ``success``, ``title``, ``description``,
        ``channel_title``, ``published_at``, ``tags``, ``view_count``,
        ``like_count``, ``duration``, ``thumbnail_url``, and ``error``.

    Examples:
        >>> import os
        >>> key = os.environ.get("YOUTUBE_API_KEY", "")
        >>> result = get_video_metadata_youtube_api.__wrapped__("dQw4w9WgXcQ", key) if key else {"success": True, "description": "skip"}
        >>> result["success"]
        True
    """
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics,contentDetails",
        "id": video_id,
        "key": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            return {
                "success": False,
                "title": "",
                "description": "",
                "channel_title": "",
                "published_at": "",
                "tags": [],
                "view_count": "",
                "like_count": "",
                "duration": "",
                "thumbnail_url": "",
                "error": "Video not found",
            }

        snippet = items[0].get("snippet", {})
        statistics = items[0].get("statistics", {})
        content_details = items[0].get("contentDetails", {})
        thumbnails = snippet.get("thumbnails", {})
        thumb_url = thumbnails.get(
            "maxres", thumbnails.get("high", thumbnails.get("default", {}))
        ).get("url", "")

        return {
            "success": True,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "tags": snippet.get("tags", []),
            "view_count": statistics.get("viewCount", ""),
            "like_count": statistics.get("likeCount", ""),
            "duration": content_details.get("duration", ""),
            "thumbnail_url": thumb_url,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "title": "",
            "description": "",
            "channel_title": "",
            "published_at": "",
            "tags": [],
            "view_count": "",
            "like_count": "",
            "duration": "",
            "thumbnail_url": "",
            "error": str(e),
        }


def ms_to_srt_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format: HH:MM:SS,mmm.

    Args:
        ms: Time in milliseconds.

    Returns:
        Formatted SRT timestamp string.

    Examples:
        >>> ms_to_srt_timestamp(0)
        '00:00:00,000'
        >>> ms_to_srt_timestamp(3661001)
        '01:01:01,001'
    """
    try:
        ms_int = int(ms)
    except Exception:
        ms_int = 0
    hours, remainder = divmod(ms_int, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def srt_timestamp_to_seconds(timestamp: str) -> int:
    """Convert an SRT timestamp to whole seconds.

    Parses timestamps in ``HH:MM:SS,mmm`` format and returns the
    equivalent time in whole seconds (truncated), suitable for use
    in YouTube ``&t=`` URL parameters.

    Args:
        timestamp: SRT-format timestamp (e.g. ``"01:02:03,456"``).

    Returns:
        Time in whole seconds.

    Examples:
        >>> srt_timestamp_to_seconds("00:00:00,000")
        0
        >>> srt_timestamp_to_seconds("00:01:30,500")
        90
        >>> srt_timestamp_to_seconds("01:01:01,001")
        3661
        >>> srt_timestamp_to_seconds("invalid")
        0
    """
    try:
        time_part = timestamp.split(",")[0]
        parts = time_part.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return 0


def serp_transcript_to_srt(transcript: Optional[List[Dict]]) -> str:
    """Convert a SerpApi-style transcript to SRT format.

    Args:
        transcript: List of dicts with ``start_ms``, ``end_ms``, ``snippet`` keys.

    Returns:
        An SRT-formatted string, or ``""`` if input is empty/invalid.

    Examples:
        >>> serp_transcript_to_srt(None)
        ''
        >>> serp_transcript_to_srt([])
        ''
        >>> serp_transcript_to_srt([{"start_ms": 0, "end_ms": 1000, "snippet": "Hi"}])
        '1\\n00:00:00,000 --> 00:00:01,000\\nHi\\n'
    """
    if not transcript or not isinstance(transcript, list):
        return ""

    srt_lines = []
    idx = 1
    for item in transcript:
        start_ms = item.get("start_ms") or item.get("start") or 0
        end_ms = item.get("end_ms") or item.get("end") or 0
        text = item.get("snippet") or item.get("text") or item.get("transcript") or ""
        text = text.replace("\n", " ").strip()
        if not text:
            continue
        start_ts = ms_to_srt_timestamp(start_ms)
        end_ts = ms_to_srt_timestamp(end_ms)
        srt_block = f"{idx}\n{start_ts} --> {end_ts}\n{text}\n"
        srt_lines.append(srt_block)
        idx += 1

    return "\n".join(srt_lines)


def render_markdown_with_timestamps(
    markdown_text: str,
    video_id: str,
    open_in_new_tab: bool = False,
    use_youtube_url: bool = False,
) -> None:
    """Render LLM markdown with clickable timestamps that seek the video.

    Timestamps matching ``H:MM:SS`` or ``HH:MM:SS`` are converted to
    clickable links. By default, links navigate within the current
    Streamlit session; optionally they can point to YouTube directly
    and/or open in a new browser tab.

    Args:
        markdown_text: The LLM-generated markdown string.
        video_id: YouTube video ID used to build the link.
        open_in_new_tab: If ``True``, links open in a new browser tab.
        use_youtube_url: If ``True``, links point to
            ``https://www.youtube.com/watch?v=…&t=…`` instead of the
            in-app ``?v=…&t=…`` relative URL.

    Examples:
        >>> render_markdown_with_timestamps("point at 0:02:30", "abc")  # doctest: +SKIP
    """
    timestamp_re = re.compile(r"\b(\d{1,2}:\d{2}:\d{2})\b")

    def _replace(match: re.Match) -> str:
        ts = match.group(1)
        seconds = srt_timestamp_to_seconds(ts)
        if use_youtube_url:
            href = f"https://www.youtube.com/watch?v={video_id}&t={seconds}"
        else:
            href = f"?v={video_id}&t={seconds}"
        target_attr = (
            ' target="_blank" rel="noopener noreferrer"' if open_in_new_tab else ""
        )
        return (
            f'<a href="{href}"{target_attr} '
            f'style="color:#1a73e8;text-decoration:underline;cursor:pointer" '
            f'title="Jump to {ts}">{ts}</a>'
        )

    processed = timestamp_re.sub(_replace, markdown_text)
    st.markdown(processed, unsafe_allow_html=True)


def copy_to_clipboard_button(text: str, label: str = "🔗 Copy share link") -> None:
    """Render a button that copies text to the clipboard on click.

    Uses ``st.html`` with inline JavaScript and the browser
    ``navigator.clipboard`` API.  The button briefly shows a
    "Copied!" confirmation before reverting to its original label.

    Args:
        text: The string to copy to the clipboard.
        label: Button label displayed to the user.

    Examples:
        >>> copy_to_clipboard_button("https://example.com")  # doctest: +SKIP
    """
    import html as html_mod

    safe_text = html_mod.escape(text, quote=True)
    safe_label = html_mod.escape(label, quote=True)
    st.html(
        f"""
        <button onclick="
            navigator.clipboard.writeText('{safe_text}');
            this.textContent='✅ Copied!';
            setTimeout(() => this.textContent='{safe_label}', 2000);
        " style="
            padding: 4px 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            background: transparent;
            cursor: pointer;
            font-size: 14px;
        ">{safe_label}</button>
        """,
        unsafe_allow_javascript=True,
    )


def hide_streamlit_chrome():
    """Hide default Streamlit UI chrome (footer badge, hamburger menu).

    Injects a ``<style>`` block that hides the Streamlit footer and main menu.
    Call once at the top of the app, after ``st.set_page_config()``.

    >>> hide_streamlit_chrome  # doctest: +ELLIPSIS
    <function hide_streamlit_chrome at ...>
    """
    from streamlit.components.v1 import html

    # html("""
    #        <script>
    #         window.top.document.querySelectorAll(`[href*="streamlit.io"]`).forEach(e => e.setAttribute("style", "display: none;"));
    #       </script>
    #     """)
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def detect_mobile_device(width_threshold: int = 600) -> bool | None:
    """Detect if the user's browser viewport is narrower than a threshold.

    Uses ``streamlit-js-eval`` to query ``window.innerWidth`` from the browser.

    Args:
        width_threshold: Pixel width at or below which the device is
            considered mobile. Defaults to 600.

    Returns:
        True if mobile, False if desktop, None if detection pending.

    Examples:
        >>> # When JS returns a phone-width viewport:
        >>> # detect_mobile_device()  -> True
        >>> # When JS hasn't responded yet:
        >>> # detect_mobile_device()  -> None
    """
    from streamlit_js_eval import streamlit_js_eval

    width = streamlit_js_eval(
        js_expressions="window.innerWidth", key="device_width_detect"
    )
    if width is None or width == 0:
        return None
    return width <= width_threshold


def is_valid_url(url_string):
    try:
        url = httpx.URL(url_string)
        return url.scheme and url.host
    except Exception:
        return False


@st.cache_data
def get_llm_icon(model_name: str, theme: str = "dark"):
    """return an LLM Icon thanks to lobe-icons
    ref: https://github.com/lobehub/lobe-icons#-cdn-usage
    """
    icon_slug = model_name.split("/")[0]
    icon_slug = icon_slug.split("-")[0] if "-" in icon_slug else icon_slug
    icon_url = (
        f"https://unpkg.com/@lobehub/icons-static-png@latest/{theme}/{icon_slug}.png"
    )
    return icon_url if is_valid_url(icon_url) else None
