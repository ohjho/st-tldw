"""Tests for get_video_metadata_oembed()."""

import pytest

from streamlit_app import get_video_metadata_oembed


# Access the unwrapped function to bypass Streamlit caching in tests.
_fetch = get_video_metadata_oembed.__wrapped__


class TestGetVideoMetadataOembed:
    """Tests for the YouTube oEmbed metadata fetcher."""

    def test_successful_fetch(self):
        """Known public video returns expected metadata fields."""
        result = _fetch("dQw4w9WgXcQ")
        assert result["success"] is True
        assert result["error"] is None
        assert result["title"]  # non-empty string
        assert result["author_name"]
        assert result["author_url"].startswith("http")
        assert result["thumbnail_url"].startswith("http")

    def test_invalid_video_id(self):
        """Bogus video ID returns a failure dict, not an exception."""
        result = _fetch("INVALID_ID_THAT_DOES_NOT_EXIST_999")
        assert result["success"] is False
        assert result["error"]  # non-empty error message
        assert result["title"] == ""

    def test_doctest(self):
        """Run the doctest embedded in the function docstring."""
        import doctest
        import streamlit_app

        results = doctest.testmod(streamlit_app, verbose=False)
        assert results.failed == 0
