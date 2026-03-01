"""Parse JSON logprobs from the OpenAI API and convert enum fields to probability distributions."""

import itertools
import math


def tokenize(logprobs_data):
    """Yield tokens and their top_logprobs from OpenAI logprobs content."""
    if logprobs_data is None:
        raise ValueError(
            "logprobs_data is None. Pass the logprobs object from the response, e.g. response.choices[0].logprobs"
        )
    if not hasattr(logprobs_data, "content"):
        raise ValueError(
            "logprobs_data has no 'content' attribute. Expected an OpenAI logprobs object (e.g. response.choices[0].logprobs)."
        )
    logprobs_sequence = logprobs_data.content
    if not logprobs_sequence:
        raise ValueError(
            "logprobs_data.content is empty. Ensure the API response was requested with logprobs=True and has content."
        )

    n = len(logprobs_sequence)
    yield logprobs_sequence[0].top_logprobs
    li = 0
    ci = 0

    def increment_ci(increase=1):
        nonlocal li, ci
        for _ in range(increase):
            if ci == len(logprobs_sequence[li].token) - 1:
                li += 1
                ci = 0
                yield logprobs_sequence[li].top_logprobs
            else:
                ci += 1

    while li < n:
        char = logprobs_sequence[li].token[ci]

        # --- Whitespace ---
        if char.isspace():
            yield from increment_ci()
            continue

        # --- Structural Tokens ---
        if char in ["{", "}", "[", "]", ":", ","]:
            yield char
            yield from increment_ci()
            continue

        # --- String Token (with JSON escape decoding: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX) ---
        if char == '"':
            yield from increment_ci()
            start_to_end_string = ""
            _escape_map = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            while li < n:
                c = logprobs_sequence[li].token[ci]
                if c == "\\":
                    yield from increment_ci()
                    if li >= n:
                        break
                    esc = logprobs_sequence[li].token[ci]
                    if esc == "u":
                        yield from increment_ci()
                        hex_str = ""
                        for _ in range(4):
                            if li < n:
                                hex_str += logprobs_sequence[li].token[ci]
                                yield from increment_ci()
                        if len(hex_str) == 4:
                            start_to_end_string += chr(int(hex_str, 16))
                    else:
                        start_to_end_string += _escape_map.get(esc, esc)
                        yield from increment_ci()
                elif c == '"':
                    break
                else:
                    start_to_end_string += c
                    yield from increment_ci()
            yield '"' + start_to_end_string + '"'
            yield from increment_ci()
            continue

        # --- Number/Keyword Tokens (incl. scientific notation: 1e-5, 2.5E+10) ---
        if char.isdigit() or char == "-":
            start_to_end_number = ""
            while li < n:
                c = logprobs_sequence[li].token[ci]
                if (
                    c.isdigit()
                    or c == "."
                    or c == "-"
                    or c in "eE+"
                ):
                    start_to_end_number += c
                    yield from increment_ci()
                else:
                    break
            yield start_to_end_number
            continue

        # --- Keywords (true, false, null) ---
        if char in "tfn":
            if logprobs_sequence[li].token[ci : ci + 4] == "true":
                yield "true"
                yield from increment_ci(4)
            elif logprobs_sequence[li].token[ci : ci + 5] == "false":
                yield "false"
                yield from increment_ci(5)
            elif logprobs_sequence[li].token[ci : ci + 4] == "null":
                yield "null"
                yield from increment_ci(4)
            else:
                raise ValueError(
                    f"Invalid keyword in logprobs stream at content index {li}, char index {ci}: "
                    f"got {logprobs_sequence[li].token!r}. Expected one of: true, false, null."
                )
            continue

        raise ValueError(
            f"Unexpected character in logprobs stream at content index {li}, char index {ci}: "
            f"{char!r} in {logprobs_sequence[li].token!r}. Expected a valid JSON token (string, number, true, false, null, or {{ }} [ ] : ,)."
        )


def get_next_token(tokens, allow_list=False):
    token = next(tokens)
    while isinstance(token, list) and not allow_list:
        token = next(tokens)
    return token


def parse_value(tokens, schema_dict):
    """Parse any JSON value (object, array, string, number, bool, null) per schema."""
    logprobs = None
    token = get_next_token(tokens, allow_list=True)
    if isinstance(token, list):
        logprobs = token
        token = get_next_token(tokens)

    if token == "{" and schema_dict.get("type") == "object":
        return parse_object(tokens, schema_dict)
    elif token == "[" and schema_dict.get("type", "array") == "array":
        return parse_array(tokens, schema_dict)

    elif token.startswith('"') and schema_dict.get("type") == "string":
        if "enum" in schema_dict:
            if logprobs is None or (isinstance(logprobs, list) and len(logprobs) == 0):
                raise ValueError(
                    "Enum field in schema but no logprobs available for this token. "
                    "Ensure the API was called with logprobs=True and top_logprobs (e.g. 20). "
                    f"Schema enum: {schema_dict['enum']}."
                )
            output = {
                class_label: sum(
                    math.exp(tlp.logprob)
                    for tlp in logprobs
                    if class_label.lower().startswith(tlp.token.strip().lower())
                )
                for class_label in schema_dict["enum"]
            }
            sum_output = sum(output.values())
            if sum_output == 0:
                raise ValueError(
                    "Enum field: no logprob mass matched any enum value. "
                    f"Schema enum: {schema_dict['enum']}. "
                    "Try enum labels with distinct prefixes, or increase top_logprobs."
                )
            output = {k: v / sum_output for k, v in output.items()}
            # Consume the rest of the string token if we only saw the opening quote
            if token == '"':
                get_next_token(tokens)
        else:
            output = token[1:-1]
        return output

    elif token == "null" and schema_dict.get("type") == "null":
        return None

    elif token in ("true", "false", "null") and schema_dict.get("type") == "boolean":
        return {"true": True, "false": False, "null": None}[token]

    elif schema_dict.get("type") == "integer":
        try:
            return int(token)
        except ValueError:
            raise ValueError(
                f"Expected an integer for schema type 'integer', got {token!r}. "
                "Check that the logprobs content matches the schema."
            ) from None
    elif schema_dict.get("type") == "number":
        try:
            if "." in token or "e" in token.lower():
                return float(token)
            return int(token)
        except ValueError:
            raise ValueError(
                f"Expected a number for schema type 'number', got {token!r}. "
                "Check that the logprobs content matches the schema."
            ) from None
    else:
        schema_type = schema_dict.get("type", "(missing)")
        raise ValueError(
            f"Unexpected token {token!r} for schema type {schema_type!r}. "
            f"Schema expects one of: object, array, string, number, integer, boolean, null. "
            "Ensure the logprobs_data content matches the JSON structure implied by the schema."
        )


def parse_object(tokens, schema_dict):
    """Parse a JSON object into a Python dict."""
    if "properties" not in schema_dict:
        raise ValueError(
            "Schema type is 'object' but schema has no 'properties'. "
            "Add a 'properties' dict, e.g. {\"type\": \"object\", \"properties\": {\"key\": {\"type\": \"string\"}}}."
        )
    properties = schema_dict["properties"]
    obj = {}
    peek = get_next_token(tokens)

    if peek == "}":
        return obj

    while True:
        if not peek.startswith('"'):
            raise ValueError(
                f"Expected a quoted string key (object property name), got {peek!r}. "
                "JSON object keys must be double-quoted strings."
            )
        # Tokenizer yields '"' then '"key"'; consume full key token when needed
        if peek == '"':
            peek = get_next_token(tokens)
        key = peek[1:-1]

        colon = get_next_token(tokens)
        if colon != ":":
            raise ValueError(
                f"Expected ':' after object key {key!r}, got {colon!r}. "
                "Check that the logprobs content is valid JSON."
            )

        if key not in properties:
            allowed = list(properties.keys())
            raise ValueError(
                f"Unknown key {key!r}. Schema only allows: {allowed}. "
                "Ensure the response matches the schema, or add this key to schema properties."
            )

        value = parse_value(tokens, properties[key])
        obj[key] = value

        separator = get_next_token(tokens)
        if separator == "}":
            return obj
        if separator != ",":
            raise ValueError(
                f"Expected ',' or '}}' after object value, got {separator!r}. "
                "Check that the logprobs content is valid JSON."
            )

        peek = get_next_token(tokens)


def parse_array(tokens, schema_dict):
    """Parse a JSON array into a Python list."""
    if "items" not in schema_dict:
        raise ValueError(
            "Schema type is 'array' but schema has no 'items'. "
            "Add an 'items' schema, e.g. {\"type\": \"array\", \"items\": {\"type\": \"string\"}}."
        )
    arr = []
    first_token = next(tokens)
    if isinstance(first_token, list):
        first_token = next(tokens)
    if first_token == "]":
        return arr

    while True:
        if arr:
            value = parse_value(tokens, schema_dict["items"])
        else:
            temp_tokens = itertools.chain([first_token], tokens)
            value = parse_value(temp_tokens, schema_dict["items"])

        arr.append(value)

        separator = next(tokens)
        if isinstance(separator, list):
            separator = next(tokens)

        if separator == "]":
            return arr
        if separator != ",":
            raise ValueError(
                f"Expected ',' or ']' after array element, got {separator!r}. "
                "Check that the logprobs content is valid JSON."
            )


def _tokenizer_wrapper(logprobs_data):
    yield from tokenize(logprobs_data)


def parse_using_schema_and_logprobs(schema_dict, logprobs_data):
    """Parse OpenAI logprobs content according to a JSON schema.

    For schema fields of type string with an \"enum\", returns a probability
    distribution over the enum values (dict mapping each enum label to a
    probability in [0, 1]) instead of the raw string. Other fields are
    parsed as usual JSON values.

    Args:
        schema_dict: JSON Schema dict (e.g. type, properties, items, enum).
        logprobs_data: OpenAI response logprobs object (e.g. response.choices[0].logprobs).

    Returns:
        Parsed structure (dict/list/primitives) with enum fields as probability dicts.
    """
    if schema_dict is None:
        raise ValueError("schema_dict is None. Pass a JSON Schema dict with at least 'type'.")
    if not isinstance(schema_dict, dict):
        raise ValueError(
            f"schema_dict must be a dict, got {type(schema_dict).__name__}. "
            "Pass a JSON Schema dict (e.g. {\"type\": \"object\", \"properties\": {...}})."
        )
    if "type" not in schema_dict:
        raise ValueError(
            "schema_dict must contain 'type'. Example: {\"type\": \"object\", \"properties\": {...}}."
        )

    tokens = _tokenizer_wrapper(logprobs_data)
    try:
        return parse_value(tokens, schema_dict)
    except StopIteration:
        raise ValueError(
            "Logprobs stream ended unexpectedly. The logprobs_data.content may not match the expected JSON structure, "
            "or the response may be truncated. Ensure logprobs=True and top_logprobs is set (e.g. 20)."
        ) from None