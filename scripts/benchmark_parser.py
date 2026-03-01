#!/usr/bin/env python3
"""Benchmark distenum parser vs json.loads on equivalent input.

Run from repo root: python scripts/benchmark_parser.py
Or: PYTHONPATH=. python scripts/benchmark_parser.py
"""

import json
import sys
import time
from pathlib import Path

# Allow importing from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from distenum import parse_using_schema_and_logprobs


# Minimal mock helpers (no pytest)
def _make_content_token(token_str, top_logprobs=None):
    if top_logprobs is None:
        top_logprobs = []
    return type("ContentToken", (), {"token": token_str, "top_logprobs": list(top_logprobs)})()


def _make_logprobs_data(content_list):
    return type("LogprobsData", (), {"content": content_list})()


def build_mock_logprobs():
    """Build logprobs content for a small JSON object (realistic multi-char tokens)."""
    content = [
        _make_content_token("{", []),
        _make_content_token('"', []),
        _make_content_token("name", []),
        _make_content_token('"', []),
        _make_content_token(":", []),
        _make_content_token('"', []),
        _make_content_token("Alice", []),
        _make_content_token('"', []),
        _make_content_token(",", []),
        _make_content_token('"', []),
        _make_content_token("count", []),
        _make_content_token('"', []),
        _make_content_token(":", []),
        _make_content_token("42", []),
        _make_content_token(",", []),
        _make_content_token('"', []),
        _make_content_token("items", []),
        _make_content_token('"', []),
        _make_content_token(":", []),
        _make_content_token("[", []),
        _make_content_token("1", []),
        _make_content_token(",", []),
        _make_content_token("2", []),
        _make_content_token(",", []),
        _make_content_token("3", []),
        _make_content_token("]", []),
        _make_content_token("}", []),
    ]
    return _make_logprobs_data(content)


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "items": {"type": "array", "items": {"type": "integer"}},
    },
}

EXPECTED = {"name": "Alice", "count": 42, "items": [1, 2, 3]}
JSON_STR = json.dumps(EXPECTED)


def run_benchmark(n=100_000):
    """Run both parsers n times and return (json_time_sec, distenum_time_sec)."""
    logprobs_data = build_mock_logprobs()

    # Warm-up
    json.loads(JSON_STR)
    parse_using_schema_and_logprobs(SCHEMA, logprobs_data)

    start = time.perf_counter()
    for _ in range(n):
        json.loads(JSON_STR)
    json_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n):
        parse_using_schema_and_logprobs(SCHEMA, logprobs_data)
    distenum_time = time.perf_counter() - start

    return json_time, distenum_time


def main():
    n = 100_000
    json_time, distenum_time = run_benchmark(n)
    ratio = distenum_time / json_time if json_time > 0 else 0

    print("Parser speed comparison (same logical structure)")
    print("=" * 55)
    print(f"  Iterations:       {n:,}")
    print(f"  json.loads:       {json_time:.3f} s  ({n / json_time:,.0f} parses/sec)  {json_time / n * 1e6:.1f} µs/parse")
    print(f"  distenum:         {distenum_time:.3f} s  ({n / distenum_time:,.0f} parses/sec)  {distenum_time / n * 1e6:.1f} µs/parse")
    print(f"  Ratio:            distenum is {ratio:.1f}× slower than json.loads")
    print()
    print("(distenum walks token-level logprobs and builds distributions for enums.)")


if __name__ == "__main__":
    main()
