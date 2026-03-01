"""Mock OpenAI logprobs structures for testing."""

import os

import pytest


def make_content_token(token_str, top_logprobs=None):
    """One content element: .token and .top_logprobs (list of objects with .token, .logprob)."""
    if top_logprobs is None:
        top_logprobs = []
    return type("ContentToken", (), {"token": token_str, "top_logprobs": list(top_logprobs)})()


def make_top_logprob(token_str, logprob):
    """One top_logprob entry."""
    return type("TopLogprob", (), {"token": token_str, "logprob": logprob})()


def make_logprobs_content(token_entries):
    """Build content list from list of (token_string, [(token, logprob), ...])."""
    content = []
    for token_str, top in token_entries:
        top_list = [make_top_logprob(t, lp) for t, lp in top]
        content.append(make_content_token(token_str, top_list))
    return content


def make_logprobs_data(content_list):
    """Build logprobs_data with .content = content_list."""
    return type("LogprobsData", (), {"content": content_list})()


@pytest.fixture(scope="module")
def openai_client():
    """OpenAI client for integration tests; skips if openai not installed or no API key."""
    openai = pytest.importorskip("openai")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    return openai.OpenAI()
