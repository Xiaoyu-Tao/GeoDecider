"""LLM client helpers.

Configuration is read from environment variables so credentials never need to be
stored in the repository:

* ``GEODECIDER_API_KEY`` (or ``DEEPSEEK_API_KEY``)
* ``GEODECIDER_BASE_URL`` (default: ``https://api.deepseek.com``)
* ``GEODECIDER_MODEL`` (default: ``deepseek-reasoner``)
"""

import os
from typing import Any, Dict, Iterable, Tuple

try:
    from .constants import FACIES_LABELS
except ImportError:  # Support ``python Facies/main.py``.
    from constants import FACIES_LABELS


def _client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for LLM calls. Install requirements.txt."
        ) from exc
    api_key = os.getenv("GEODECIDER_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEODECIDER_API_KEY or DEEPSEEK_API_KEY."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("GEODECIDER_BASE_URL", "https://api.deepseek.com"),
    )


def _usage_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def chat_completion(
    messages: Iterable[Dict[str, str]], *, json_object: bool = False
) -> Tuple[str, str, Dict[str, Any]]:
    request: Dict[str, Any] = {
        "model": os.getenv("GEODECIDER_MODEL", "deepseek-reasoner"),
        "messages": list(messages),
        "stream": False,
        "extra_body": {"enable_thinking": True},
    }
    if json_object:
        request["response_format"] = {"type": "json_object"}

    response = _client().chat.completions.create(**request)
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or ""
    content = message.content or ""
    metadata = {
        "model": getattr(response, "model", request["model"]),
        "usage": _usage_dict(getattr(response, "usage", None)),
    }
    return reasoning, content, metadata


def get_json_result(content: str, system_prompt: str):
    return chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        json_object=True,
    )


def get_result(content: str):
    labels = ", ".join(FACIES_LABELS)
    system_prompt = (
        "Classify every input row. Return only a JSON object in the form "
        '{"answer": ["X1", "X2", ...]}. Each Xi must be one of: '
        f"{labels}. The answer length must exactly match the number of input rows."
    )
    return get_json_result(content, system_prompt)


def get_result_trend(content: str):
    return chat_completion([{"role": "user", "content": content}])
