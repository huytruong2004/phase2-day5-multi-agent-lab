"""LLM client abstraction.

Single retry on transient errors, hard per-call timeout, deterministic cost
accounting from a static price table. Agents depend on this interface, not the
SDK directly.
"""

from dataclasses import dataclass

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from multi_agent_research_lab.core.config import get_settings

_PRICE_PER_1K_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4-turbo": (0.01, 0.03),
}

_REQUEST_TIMEOUT_SECONDS = 30.0
_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = _PRICE_PER_1K_TOKENS_USD.get(model)
    if prices is None:
        return None
    in_price, out_price = prices
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price


class LLMClient:
    """OpenAI-backed LLM client. One retry on transient errors, fail fast otherwise."""

    def __init__(self, model: str | None = None, temperature: float = 0.2) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot create LLMClient.")
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=_REQUEST_TIMEOUT_SECONDS)
        self._model = model or settings.openai_model
        self._temperature = temperature

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1.0),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        reraise=True,
    )
    def _call(self, system_prompt: str, user_prompt: str, max_tokens: int):
        return self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 800,
    ) -> LLMResponse:
        response = self._call(system_prompt, user_prompt, max_tokens)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else None
        out_tok = usage.completion_tokens if usage else None
        cost = (
            _estimate_cost(self._model, in_tok, out_tok)
            if in_tok is not None and out_tok is not None
            else None
        )
        return LLMResponse(
            content=content,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model=self._model,
        )
