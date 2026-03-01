"""Tests for distenum parser and enum distribution."""

import pytest
from distenum import parse_using_schema_and_logprobs, tokenize

from tests.conftest import (
    make_content_token,
    make_logprobs_data,
    make_top_logprob,
)

# Schema and request helpers for OpenAI API integration tests
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
        }
    },
    "required": ["sentiment"],
    "additionalProperties": False,
}


def _create_completion(client, schema, user_content, **kwargs):
    """Call chat completions with JSON schema and logprobs."""
    defaults = {
        "model": "gpt-4o-2024-08-06",
        "messages": [{"role": "user", "content": user_content}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "test_schema", "strict": True, "schema": schema},
        },
        "temperature": 0,
        "max_tokens": 256,
        "logprobs": True,
        "top_logprobs": 20,
    }
    defaults.update(kwargs)
    return client.chat.completions.create(**defaults)


def test_empty_object():
    """Parse empty object {}."""
    # content: '{', '}'
    content = [
        make_content_token("{", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result == {}


def test_string_enum_distribution():
    """Enum string returns a probability distribution that sums to 1."""
    # Realistic tokens: key "label", value "positive" (multi-char tokens like the API returns)
    top = [
        make_top_logprob("pos", -0.5),
        make_top_logprob("positive", -0.2),
        make_top_logprob("negative", -2.0),
        make_top_logprob("neutral", -3.0),
    ]
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("label", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', top),
        make_content_token("positive", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["positive", "negative", "neutral"]}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "label" in result
    dist = result["label"]
    assert set(dist.keys()) == {"positive", "negative", "neutral"}
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["positive"] > dist["negative"]
    assert dist["positive"] > dist["neutral"]


def test_enum_distribution_sums_to_one():
    """With only one matching logprob, distribution still normalizes."""
    top = [make_top_logprob("pos", -0.1)]
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("sentiment", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', top),
        make_content_token("pos", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"sentiment": {"type": "string", "enum": ["pos", "neg"]}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    dist = result["sentiment"]
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert dist["pos"] > 0
    assert dist["neg"] == 0.0


def test_schema_type_null():
    """Schema with type \"null\" parses null token as None."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("optional", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("null", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"optional": {"type": "null"}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["optional"] is None


def test_string_escape_sequences():
    """String escape sequences (\\n, \\\", \\uXXXX, etc.) are decoded."""
    # Realistic tokens: key "msg", value "hello" + \n + "world"
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("msg", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', []),
        make_content_token("hello", []),
        make_content_token("\\n", []),
        make_content_token("world", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["msg"] == "hello\nworld"


def test_number_scientific_notation():
    """Numbers in scientific notation (e.g. 1e-5, 2.5E+10) are tokenized and parsed."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("rate", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("1e-5", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"rate": {"type": "number"}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["rate"] == 1e-5


# --- Helpful error messages ---


def test_error_logprobs_data_none():
    """Clear error when logprobs_data is None."""
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"type": "object", "properties": {}}, None)
    assert "logprobs_data is None" in str(exc_info.value)
    assert "response.choices[0].logprobs" in str(exc_info.value)


def test_error_logprobs_data_no_content():
    """Clear error when logprobs_data has no content attribute."""
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"type": "object", "properties": {}}, object())
    assert "content" in str(exc_info.value)


def test_error_schema_dict_none():
    """Clear error when schema_dict is None."""
    logprobs_data = make_logprobs_data([make_content_token("{", []), make_content_token("}", [])])
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(None, logprobs_data)
    assert "schema_dict is None" in str(exc_info.value)


def test_error_schema_dict_missing_type():
    """Clear error when schema_dict has no 'type'."""
    logprobs_data = make_logprobs_data([make_content_token("{", []), make_content_token("}", [])])
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"properties": {}}, logprobs_data)
    assert "type" in str(exc_info.value)


def test_error_object_schema_missing_properties():
    """Clear error when object schema has no 'properties'."""
    logprobs_data = make_logprobs_data([make_content_token("{", []), make_content_token("}", [])])
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"type": "object"}, logprobs_data)
    assert "properties" in str(exc_info.value)


def test_error_unknown_object_key():
    """Clear error when response has a key not in schema."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("extra_key", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("1", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"allowed_key": {"type": "integer"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "Unknown key" in str(exc_info.value) or "extra_key" in str(exc_info.value)
    assert "allowed_key" in str(exc_info.value)


def test_error_enum_no_logprobs():
    """Clear error when enum field has no logprobs (empty list)."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("s", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', []),  # opening quote with empty top_logprobs
        make_content_token("a", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    # Use a logprobs_data where the content element for '"' has empty top_logprobs
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"s": {"type": "string", "enum": ["a", "b"]}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "enum" in str(exc_info.value).lower() or "logprobs" in str(exc_info.value).lower()


# --- Additional coverage: tokenizer, validation, types, escapes, error paths ---


def test_error_logprobs_data_content_empty():
    """Clear error when logprobs_data.content is empty list."""
    empty_content = make_logprobs_data([])
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"type": "object", "properties": {}}, empty_content)
    assert "empty" in str(exc_info.value).lower()


def test_error_tokenize_invalid_keyword():
    """Tokenizer raises when token looks like keyword but isn't true/false/null."""
    # "tru" or "nope" - need content that spells "tru" so tokenizer sees t,r,u and fails
    content = [make_content_token("tru", [])]
    logprobs_data = make_logprobs_data(content)
    with pytest.raises(ValueError) as exc_info:
        list(tokenize(logprobs_data))
    assert "true" in str(exc_info.value).lower() or "false" in str(exc_info.value).lower() or "null" in str(exc_info.value).lower()


def test_error_tokenize_unexpected_character():
    """Tokenizer raises for invalid JSON character."""
    content = [make_content_token("#", [])]
    logprobs_data = make_logprobs_data(content)
    with pytest.raises(ValueError) as exc_info:
        list(tokenize(logprobs_data))
    assert "Unexpected" in str(exc_info.value) or "JSON" in str(exc_info.value)


def test_error_schema_dict_not_dict():
    """Clear error when schema_dict is not a dict (e.g. list)."""
    logprobs_data = make_logprobs_data([make_content_token("{", []), make_content_token("}", [])])
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs([], logprobs_data)
    assert "dict" in str(exc_info.value).lower()


def test_error_array_schema_missing_items():
    """Clear error when array schema has no 'items'."""
    content = [
        make_content_token("[", []),
        make_content_token("]", []),
    ]
    logprobs_data = make_logprobs_data(content)
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs({"type": "array"}, logprobs_data)
    assert "items" in str(exc_info.value).lower()


def test_boolean_type_true_and_false():
    """Schema type boolean parses true and false."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("enabled", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("true", []),
        make_content_token(",", []),
        make_content_token('"', []),
        make_content_token("visible", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("false", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {
        "type": "object",
        "properties": {"enabled": {"type": "boolean"}, "visible": {"type": "boolean"}},
    }
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["enabled"] is True
    assert result["visible"] is False


def test_integer_type():
    """Schema type integer parses integer tokens."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("count", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("42", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["count"] == 42


def test_non_enum_string():
    """Non-enum string is returned as decoded string value."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("text", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', []),
        make_content_token("hi", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"text": {"type": "string"}}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["text"] == "hi"


def test_empty_array():
    """Empty array [] parses to empty list."""
    content = [
        make_content_token("[", []),
        make_content_token("]", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "array", "items": {"type": "integer"}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result == []


def test_array_of_primitives():
    """Array of numbers parses to list of numbers."""
    content = [
        make_content_token("[", []),
        make_content_token("100", []),
        make_content_token(",", []),
        make_content_token("200", []),
        make_content_token(",", []),
        make_content_token("300", []),
        make_content_token("]", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "array", "items": {"type": "integer"}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result == [100, 200, 300]


def test_string_escape_unicode():
    """Unicode escape \\uXXXX is decoded."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("letter", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', []),
        make_content_token("\\u0041", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"letter": {"type": "string"}}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["letter"] == "A"


def test_string_escape_quote_and_backslash():
    """Escapes \\\" and \\\\ are decoded in strings."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("payload", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', []),
        make_content_token('\\"', []),
        make_content_token("a", []),
        make_content_token("\\\\", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"payload": {"type": "string"}}}
    result = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert result["payload"] == '"a\\'


def test_error_stop_iteration_stream_ended():
    """Truncated logprobs stream raises a sensible error (ValueError or IndexError)."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("key", []),
        make_content_token('"', []),
        make_content_token(":", []),
        # missing value and }
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    with pytest.raises((ValueError, IndexError)) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    msg = str(exc_info.value).lower()
    assert "ended" in msg or "unexpected" in msg or "index" in msg


def test_error_enum_zero_mass():
    """Clear error when no logprob mass matches any enum value."""
    top = [make_top_logprob("x", -0.1), make_top_logprob("y", -0.2)]
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("value", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token('"', top),
        make_content_token("z", []),
        make_content_token('"', []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"value": {"type": "string", "enum": ["a", "b"]}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "no logprob mass" in str(exc_info.value).lower() or "matched" in str(exc_info.value).lower()


def test_error_invalid_integer_token():
    """Clear error when schema expects integer but token is not valid."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("number", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("abc", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"number": {"type": "integer"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    msg = str(exc_info.value).lower()
    assert "integer" in msg or "unexpected" in msg or "token" in msg


def test_error_invalid_number_token():
    """Clear error when schema expects number but token is not valid."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("value", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("xyz", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"value": {"type": "number"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "number" in str(exc_info.value).lower()


def test_error_unexpected_token_for_schema_type():
    """Clear error when token doesn't match schema type (e.g. number where string expected)."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("score", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("42", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"score": {"type": "string"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "Unexpected token" in str(exc_info.value) or "string" in str(exc_info.value).lower()


def test_error_object_expected_colon_after_key():
    """Clear error when ':' is missing after object key."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("key", []),
        make_content_token('"', []),
        make_content_token(",", []),  # wrong: should be :
        make_content_token("1", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"key": {"type": "integer"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert ":" in str(exc_info.value)


def test_error_object_expected_comma_or_brace_after_value():
    """Clear error when ',' or '}' is wrong after object value."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("key", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("1", []),
        make_content_token("]", []),  # wrong: should be } or ,
        make_content_token("]", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"key": {"type": "integer"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "," in str(exc_info.value) or "}" in str(exc_info.value)


def test_error_array_expected_comma_or_bracket():
    """Clear error when ',' or ']' is wrong after array element."""
    content = [
        make_content_token("[", []),
        make_content_token("1", []),
        make_content_token("}", []),  # wrong
        make_content_token("]", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "array", "items": {"type": "integer"}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "," in str(exc_info.value) or "]" in str(exc_info.value)


def test_error_object_key_not_string():
    """Clear error when object key is not a quoted string (e.g. number)."""
    content = [
        make_content_token("{", []),
        make_content_token("42", []),  # key should be quoted string
        make_content_token(":", []),
        make_content_token("2", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    schema = {"type": "object", "properties": {"42": {"type": "integer"}}}
    with pytest.raises(ValueError) as exc_info:
        parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "key" in str(exc_info.value).lower() or "string" in str(exc_info.value).lower()


def test_tokenize_yields_expected_sequence():
    """tokenize() yields list of logprobs then structural/string tokens for simple JSON."""
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("key", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("1", []),
        make_content_token("}", []),
    ]
    logprobs_data = make_logprobs_data(content)
    gen = tokenize(logprobs_data)
    # Don't exhaust generator (tokenizer can raise IndexError at end). Take first 12 yields only.
    tokens = [next(gen) for _ in range(12)]
    assert tokens[0] == []
    assert tokens[1] == "{"
    assert ":" in tokens
    assert "1" in tokens
    assert any(t == '"key"' for t in tokens if isinstance(t, str))


def test_tokenize_keyword_at_end_of_stream_no_index_error():
    """Tokenizer does not IndexError when last content token is a keyword (true/false/null)."""
    # Content ending with "null" and no token after it — increment_ci(4) would advance past end
    content = [
        make_content_token("{", []),
        make_content_token('"', []),
        make_content_token("x", []),
        make_content_token('"', []),
        make_content_token(":", []),
        make_content_token("null", []),
    ]
    logprobs_data = make_logprobs_data(content)
    # Consuming all tokenizer output must not raise IndexError
    tokens = list(tokenize(logprobs_data))
    assert "null" in tokens


# --- OpenAI API integration tests (run only when openai is available and OPENAI_API_KEY is set) ---


def test_openai_single_enum_returns_distribution(openai_client):
    """Call API for a single enum field; parsed result has a distribution with correct keys."""
    response = _create_completion(
        openai_client,
        SENTIMENT_SCHEMA,
        "Reply with exactly one word: positive, negative, or neutral. Choose positive.",
    )
    logprobs_data = response.choices[0].logprobs
    parsed = parse_using_schema_and_logprobs(SENTIMENT_SCHEMA, logprobs_data)
    assert "sentiment" in parsed
    dist = parsed["sentiment"]
    assert set(dist.keys()) == {"positive", "negative", "neutral"}
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert dist["positive"] > 0


def test_openai_enum_distribution_sums_to_one(openai_client):
    """Call API with enum; parsed distribution over enum values sums to 1."""
    response = _create_completion(
        openai_client,
        SENTIMENT_SCHEMA,
        "Say only: negative",
    )
    logprobs_data = response.choices[0].logprobs
    parsed = parse_using_schema_and_logprobs(SENTIMENT_SCHEMA, logprobs_data)
    dist = parsed["sentiment"]
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_openai_nested_array_of_objects_with_enum(openai_client):
    """Call API with array of objects containing enum; each enum field is a distribution."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "sentiment": {
                            "type": "string",
                            "enum": ["positive", "negative", "neutral"],
                        },
                    },
                    "required": ["text", "sentiment"],
                    "additionalProperties": False,
                },
                "minItems": 2,
                "maxItems": 2,
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    response = _create_completion(
        openai_client,
        schema,
        "Return a JSON object with key 'items': an array of exactly 2 objects. "
        "Each object has 'text' (one short sentence) and 'sentiment' (one of: positive, negative, neutral).",
        max_tokens=512,
    )
    logprobs_data = response.choices[0].logprobs
    parsed = parse_using_schema_and_logprobs(schema, logprobs_data)
    assert "items" in parsed
    items = parsed["items"]
    assert len(items) == 2
    for item in items:
        assert "text" in item
        assert "sentiment" in item
        dist = item["sentiment"]
        assert set(dist.keys()) == {"positive", "negative", "neutral"}
        assert abs(sum(dist.values()) - 1.0) < 1e-6
