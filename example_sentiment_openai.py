"""Example: call OpenAI with JSON schema + logprobs and parse enum distributions."""

import json

from distenum import parse_using_schema_and_logprobs

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Install with: pip install distenum[openai]")

client = OpenAI()

schema_dict = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "A list of text sentiment dictionaries.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to be analyzed for sentiment.",
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "The sentiment label for the text.",
                        "enum": ["positive", "negative"],
                    },
                },
                "required": ["text", "sentiment"],
                "additionalProperties": False,
            },
            "minItems": 10,
            "maxItems": 10,
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "user",
            "content": "Generate 10 random sentences where sentiment is hard to determine and provide their sentiment (positive, negative).",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_text_list",
            "strict": True,
            "schema": schema_dict,
        },
    },
    temperature=1,
    max_tokens=8192,
    top_p=1,
    logprobs=True,
    top_logprobs=20,
)

result = json.loads(response.choices[0].message.content)
logprobs_data = response.choices[0].logprobs

print("Result (without classification probabilities):", result)
print("Result (with classification probabilities):", parse_using_schema_and_logprobs(schema_dict, logprobs_data))
