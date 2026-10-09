"""Tests for deepseek provider module.

Tests cover:
- Bug 10: call_with_logit_bias should use retry logic
"""

import pytest
from unittest.mock import patch, MagicMock


class TestBaseUrlNormalization:
    """base_url should yield a correct endpoint whether or not it includes /v1.

    deepseek.py appends '/v1/chat/completions'. A caller that passes
    'https://api.deepseek.com/v1' previously produced a doubled '/v1/v1/...'
    path and a 404.
    """

    def _make(self, base_url):
        from src.llm.deepseek import DeepSeekProvider
        from src.llm.provider import LLMProviderConfig

        cfg = LLMProviderConfig(api_key="k", base_url=base_url, model="deepseek-chat")
        return DeepSeekProvider(cfg)

    def test_endpoint_without_v1(self):
        p = self._make("https://api.deepseek.com")
        assert p._chat_url() == "https://api.deepseek.com/v1/chat/completions"

    def test_endpoint_with_v1_not_doubled(self):
        p = self._make("https://api.deepseek.com/v1")
        assert p._chat_url() == "https://api.deepseek.com/v1/chat/completions"

    def test_endpoint_with_trailing_slash(self):
        p = self._make("https://api.deepseek.com/v1/")
        assert p._chat_url() == "https://api.deepseek.com/v1/chat/completions"


class TestLogitBiasRetry:
    """Tests for call_with_logit_bias retry behavior (Bug 10)."""

    def test_call_with_logit_bias_retries_on_rate_limit(self):
        """call_with_logit_bias should retry on rate limit errors."""
        from src.llm.deepseek import DeepSeekProvider
        from src.llm.provider import LLMRateLimitError, LLMResponse

        provider = DeepSeekProvider.__new__(DeepSeekProvider)
        provider.config = MagicMock()
        provider.config.model = "deepseek-chat"
        provider.base_url = "https://api.deepseek.com"
        # provider_name is a property on DeepSeekProvider, no need to set it
        provider.retry_config = {
            "max_retries": 3,
            "base_delay": 0.01,
            "max_delay": 0.1,
        }
        provider._total_input_tokens = 0
        provider._total_output_tokens = 0
        provider._total_calls = 0

        # Mock _call_api to fail once then succeed
        call_count = 0
        def mock_call_api(messages, temperature=None, max_tokens=None,
                          require_json=False, logit_bias=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMRateLimitError("Rate limited")
            return LLMResponse(
                content="Generated text",
                model="deepseek-chat",
                input_tokens=10,
                output_tokens=20,
            )

        provider._call_api = mock_call_api

        result = provider.call_with_logit_bias(
            system_prompt="System",
            user_prompt="User",
            logit_bias={"123": -100},
            temperature=0.7,
        )

        assert result == "Generated text"
        assert call_count == 2  # First failed, second succeeded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestFinishReason:
    def _provider(self):
        from src.llm.deepseek import DeepSeekProvider
        p = DeepSeekProvider.__new__(DeepSeekProvider)
        p.config = MagicMock()
        p.config.model = "deepseek-chat"
        p.config.timeout = 5
        p.config.temperature = 0.5
        p.config.max_tokens = 100
        p.base_url = "https://api.deepseek.com"
        p.headers = {}
        p.retry_config = {"max_retries": 2, "base_delay": 0.0, "max_delay": 0.0}
        p._total_input_tokens = p._total_output_tokens = p._total_calls = 0
        return p

    def _reply(self, finish_reason):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": "partial"}, "finish_reason": finish_reason}],
            "usage": {},
        }
        return resp

    def test_finish_reason_is_recorded(self):
        p = self._provider()
        with patch("src.llm.deepseek.requests.post", return_value=self._reply("length")):
            r = p._call_api([])
        assert r.finish_reason == "length"

    def test_truncated_reply_rejected_when_asked(self):
        from src.llm.provider import LLMTruncatedError
        p = self._provider()
        with patch("src.llm.deepseek.requests.post", return_value=self._reply("length")) as post:
            with pytest.raises(LLMTruncatedError):
                p.call("s", "u", reject_truncated=True)
        assert post.call_count == 1  # truncation is deterministic: no retry

    def test_truncated_reply_returned_by_default(self):
        p = self._provider()
        with patch("src.llm.deepseek.requests.post", return_value=self._reply("length")):
            assert p.call("s", "u") == "partial"

    def test_complete_reply_passes(self):
        p = self._provider()
        with patch("src.llm.deepseek.requests.post", return_value=self._reply("stop")):
            assert p.call("s", "u", reject_truncated=True) == "partial"
