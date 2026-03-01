# distenum

**distenum** parses JSON outputs with logprobs from the OpenAI API and converts `enum`-type string fields into probability distributions instead of a single label.

## What it does

With structured outputs, the API returns a single enum value (e.g. `"positive"`). If you request `logprobs=True`, you also get token-level logprobs. **distenum** turns those logprobs into a probability distribution over your enum options so you can see how confident the model was in each choice.

**Example:** For a sentiment field with `enum: ["positive", "negative", "neutral"]`:

| From the API (content only) | With distenum (using logprobs) |
|-----------------------------|---------------------------------|
| `"sentiment": "positive"`   | `"sentiment": {"positive": 0.72, "negative": 0.18, "neutral": 0.10}` |

So instead of a single label, you get a distribution you can use for uncertainty, ranking, or thresholding (e.g. only accept when `positive` probability > 0.8).

## Install

```bash
pip install distenum
```

To run the example script (calls the OpenAI API):

```bash
pip install distenum[openai]
```

## Quick start

Install the package and the OpenAI client: `pip install distenum[openai]`. Then call the API with `logprobs=True` and `top_logprobs=20`, and pass the response logprobs into distenum:

```python
from openai import OpenAI
from distenum import parse_using_schema_and_logprobs

schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        }
    }
}

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Your prompt"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "my_schema", "strict": True, "schema": schema}
    },
    logprobs=True,
    top_logprobs=20,
)

logprobs_data = response.choices[0].logprobs
parsed = parse_using_schema_and_logprobs(schema, logprobs_data)
# parsed["sentiment"] might be: {"positive": 0.72, "negative": 0.18, "neutral": 0.10}
```

## Enum design tips

- **Different prefixes:** Enum values are matched to token logprobs by **prefix**. Prefer enum labels that do not share a common prefix (e.g. `"positive"`, `"negative"`, `"neutral"` are good; `"pos"` and `"positive"` can blur probabilities).
- **Fewer is better:** The API returns at most **20** logprobs per token (`top_logprobs=20`). With many enum values, most will get no mass; keep enums small for meaningful distributions.

## API

- **`parse_using_schema_and_logprobs(schema_dict, logprobs_data)`**  
  Parses the logprobs stream according to the JSON Schema. Fields of type `string` with an `enum` are returned as a dict mapping each enum label to a probability (non-negative, summing to 1). Other fields are parsed as normal JSON values.

- **`tokenize(logprobs_data)`**  
  Low-level generator that yields tokens and their top-logprobs from the OpenAI logprobs content.

## Performance

The parser walks token-level logprobs and builds probability distributions for enum fields, so it is slower than parsing the same JSON with the standard library. A rough comparison (same logical structure, 100k iterations):

| Parser        | Time (100k parses) | Throughput   | Avg per parse |
|---------------|--------------------|--------------|---------------|
| `json.loads`  | ~0.16 s            | ~630k/sec    | ~1.6 µs       |
| distenum      | ~3.0 s             | ~33k/sec     | ~30 µs        |

So distenum is typically **about 15–20× slower** than `json.loads` for the same structure. In absolute terms, **~30 µs per parse** is negligible compared to an OpenAI API call (typically hundreds of milliseconds to several seconds). Parsing a single response adds no meaningful latency.

To run the benchmark yourself from the repo root:

```bash
PYTHONPATH=. python scripts/benchmark_parser.py
```

## Example script

From the repo root (with `distenum[openai]` installed):

```bash
python example_sentiment_openai.py
```

## License

MIT
