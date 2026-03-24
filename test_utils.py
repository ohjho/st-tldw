"""Unit tests for utils.py — pure function tests (no network calls)."""

from unittest.mock import patch

import pytest

from utils import extract_video_id, ms_to_srt_timestamp, serp_transcript_to_srt, srt_timestamp_to_seconds


class TestExtractVideoId:
    """Tests for extract_video_id()."""

    def test_standard_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120") == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        assert extract_video_id("not-a-valid-url") is None

    def test_empty_string(self):
        assert extract_video_id("") is None


class TestMsToSrtTimestamp:
    """Tests for ms_to_srt_timestamp()."""

    def test_zero(self):
        assert ms_to_srt_timestamp(0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert ms_to_srt_timestamp(5000) == "00:00:05,000"

    def test_full_format(self):
        # 1h 1m 1s 1ms = 3_661_001 ms
        assert ms_to_srt_timestamp(3_661_001) == "01:01:01,001"

    def test_milliseconds_only(self):
        assert ms_to_srt_timestamp(123) == "00:00:00,123"

    def test_non_int_input(self):
        """Non-numeric input falls back to 00:00:00,000."""
        assert ms_to_srt_timestamp("abc") == "00:00:00,000"

    def test_float_input(self):
        """Float input is truncated to int."""
        assert ms_to_srt_timestamp(1500.7) == "00:00:01,500"


class TestSrtTimestampToSeconds:
    """Tests for srt_timestamp_to_seconds()."""

    def test_zero(self):
        assert srt_timestamp_to_seconds("00:00:00,000") == 0

    def test_simple_seconds(self):
        assert srt_timestamp_to_seconds("00:00:05,000") == 5

    def test_minutes_and_seconds(self):
        assert srt_timestamp_to_seconds("00:01:30,500") == 90

    def test_full_format(self):
        assert srt_timestamp_to_seconds("01:01:01,001") == 3661

    def test_milliseconds_truncated(self):
        """Milliseconds are discarded, not rounded."""
        assert srt_timestamp_to_seconds("00:00:01,999") == 1

    def test_no_milliseconds(self):
        """Timestamps without the comma+ms portion still parse."""
        assert srt_timestamp_to_seconds("00:02:00") == 120

    def test_invalid_input(self):
        assert srt_timestamp_to_seconds("invalid") == 0

    def test_empty_string(self):
        assert srt_timestamp_to_seconds("") == 0


class TestSerpTranscriptToSrt:
    """Tests for serp_transcript_to_srt()."""

    def test_basic_conversion(self):
        transcript = [
            {"start_ms": 0, "end_ms": 1000, "snippet": "Hello"},
            {"start_ms": 1000, "end_ms": 2000, "snippet": "World"},
        ]
        result = serp_transcript_to_srt(transcript)
        assert "1\n00:00:00,000 --> 00:00:01,000\nHello\n" in result
        assert "2\n00:00:01,000 --> 00:00:02,000\nWorld\n" in result

    def test_empty_list(self):
        assert serp_transcript_to_srt([]) == ""

    def test_none_input(self):
        assert serp_transcript_to_srt(None) == ""

    def test_non_list_input(self):
        assert serp_transcript_to_srt("not a list") == ""

    def test_missing_fields(self):
        """Items with missing keys use fallback values."""
        transcript = [{"snippet": "Only text"}]
        result = serp_transcript_to_srt(transcript)
        assert "1\n00:00:00,000 --> 00:00:00,000\nOnly text\n" in result

    def test_skips_empty_text(self):
        """Entries with empty/missing text are skipped."""
        transcript = [
            {"start_ms": 0, "end_ms": 1000, "snippet": ""},
            {"start_ms": 1000, "end_ms": 2000, "snippet": "Kept"},
        ]
        result = serp_transcript_to_srt(transcript)
        assert result.startswith("1\n")
        assert "Kept" in result

    def test_alternative_keys(self):
        """Supports 'start'/'end'/'text' as alternative keys."""
        transcript = [{"start": 500, "end": 1500, "text": "Alt keys"}]
        result = serp_transcript_to_srt(transcript)
        assert "00:00:00,500 --> 00:00:01,500" in result
        assert "Alt keys" in result

    def test_newlines_in_text_replaced(self):
        """Newlines within snippet text are replaced with spaces."""
        transcript = [{"start_ms": 0, "end_ms": 1000, "snippet": "Line1\nLine2"}]
        result = serp_transcript_to_srt(transcript)
        assert "Line1 Line2" in result
        # Only structural newlines should remain
        assert "Line1\nLine2" not in result


class TestDetectMobileDevice:
    """Tests for detect_mobile_device()."""

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_phone_width(self, mock_js_eval):
        """Phone-width viewport (375px) returns True."""
        mock_js_eval.return_value = 375
        from utils import detect_mobile_device

        assert detect_mobile_device() is True

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_desktop_width(self, mock_js_eval):
        """Desktop-width viewport (1024px) returns False."""
        mock_js_eval.return_value = 1024
        from utils import detect_mobile_device

        assert detect_mobile_device() is False

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_none_pending(self, mock_js_eval):
        """JS bridge not yet responded returns None."""
        mock_js_eval.return_value = None
        from utils import detect_mobile_device

        assert detect_mobile_device() is None

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_zero_quirk(self, mock_js_eval):
        """JS returning 0 (known quirk) returns None."""
        mock_js_eval.return_value = 0
        from utils import detect_mobile_device

        assert detect_mobile_device() is None

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_boundary(self, mock_js_eval):
        """Exactly at threshold (600px) returns True."""
        mock_js_eval.return_value = 600
        from utils import detect_mobile_device

        assert detect_mobile_device() is True

    @patch("streamlit_js_eval.streamlit_js_eval")
    def test_custom_threshold(self, mock_js_eval):
        """Custom threshold is respected."""
        mock_js_eval.return_value = 500
        from utils import detect_mobile_device

        assert detect_mobile_device(width_threshold=480) is False
        assert detect_mobile_device(width_threshold=500) is True
