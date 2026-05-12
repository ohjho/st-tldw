"""Integration tests for the Ollama Cloud LLM connection."""

import os

import litellm
import pytest
from dotenv import load_dotenv

if os.path.isfile("./secrets.env"):
    load_dotenv("./secrets.env")

from utils import get_ollama_free_models  # noqa: E402

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_API_BASE = "https://ollama.com"

# Free-tier model used as the default test target. `gpt-oss:20b` is
# available on the Ollama Cloud free tier; `ollama_chat/` routes to the
# chat endpoint (the bare `ollama/` prefix targets /api/generate and
# 403s on Ollama Cloud).
DEFAULT_OLLAMA_MODEL = "ollama_chat/gpt-oss:20b"


@pytest.mark.skipif(not OLLAMA_API_KEY, reason="OLLAMA_API_KEY not set")
class TestOllamaCloudConnection:
    """Live calls to Ollama Cloud via litellm."""

    def test_default_model_is_callable(self):
        """The default Ollama model returns a non-empty completion.

        ``max_tokens`` is set generously because gpt-oss models spend
        tokens on a hidden ``<think>`` phase before producing visible
        content; a tight budget can yield an empty string even when the
        request succeeds.
        """
        response = litellm.completion(
            model=DEFAULT_OLLAMA_MODEL,
            messages=[
                {"role": "user", "content": "Reply with the single word: pong"}
            ],
            api_key=OLLAMA_API_KEY,
            api_base=OLLAMA_API_BASE,
            temperature=0.0,
            max_tokens=200,
        )
        content = response["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert content.strip()


class TestGetOllamaFreeModels:
    """Tests for the free-tier model discovery helper."""

    def test_returns_nonempty_list(self):
        """Fetch returns at least one free-tier model name."""
        # Bypass @st.cache_data for deterministic test runs.
        free = get_ollama_free_models.__wrapped__()
        assert isinstance(free, list)
        assert len(free) > 0
        assert all(isinstance(name, str) and name for name in free)

    def test_default_model_is_in_free_list(self):
        """The model used by the test suite is itself reported as free."""
        free = get_ollama_free_models.__wrapped__()
        bare_name = DEFAULT_OLLAMA_MODEL.split("/", 1)[1]
        assert bare_name in free


class TestOllamaCloudNoKey:
    """Tests that don't require a valid key."""

    def test_invalid_key_raises(self):
        """An invalid API key surfaces as a litellm exception."""
        with pytest.raises(Exception):
            litellm.completion(
                model=DEFAULT_OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_key="INVALID_KEY",
                api_base=OLLAMA_API_BASE,
                temperature=0.0,
                max_tokens=5,
            )
