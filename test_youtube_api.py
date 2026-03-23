"""Tests for get_video_metadata_youtube_api()."""

import os

import pytest

from utils import get_video_metadata_youtube_api

# Access the unwrapped function to bypass Streamlit caching in tests.
_fetch = get_video_metadata_youtube_api.__wrapped__

API_KEY = os.environ.get("YOUTUBE_API_KEY", "")


@pytest.mark.skipif(not API_KEY, reason="YOUTUBE_API_KEY not set")
class TestGetVideoMetadataYoutubeApi:
    """Tests for the YouTube Data API metadata fetcher."""

    def test_successful_fetch(self):
        """Known public video returns expected metadata fields."""
        result = _fetch("dQw4w9WgXcQ", API_KEY)
        assert result["success"] is True
        assert result["error"] is None
        assert result["title"]  # non-empty string
        assert result["description"]  # non-empty string
        assert result["channel_title"]
        assert result["published_at"]
        assert isinstance(result["tags"], list)
        assert result["view_count"]
        assert result["duration"]
        assert result["thumbnail_url"].startswith("http")

    def test_invalid_video_id(self):
        """Bogus video ID returns a failure dict, not an exception."""
        result = _fetch("INVALID_ID_THAT_DOES_NOT_EXIST_999", API_KEY)
        assert result["success"] is False
        assert result["error"]  # non-empty error message
        assert result["description"] == ""
        assert result["tags"] == []


class TestYoutubeApiNoKey:
    """Tests that work without an API key."""

    def test_invalid_key_returns_error(self):
        """An invalid API key returns a failure dict."""
        result = _fetch("dQw4w9WgXcQ", "INVALID_KEY")
        assert result["success"] is False
        assert result["error"]
