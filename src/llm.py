import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class _OllamaResponse:
    """Mimics Groq's response structure so callers can use .choices[0].message.content and .usage."""

    class Choice:
        class Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self.Message(content)

    class Usage:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    def __init__(self, content: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.choices = [self.Choice(content)]
        self.usage = self.Usage(prompt_tokens, completion_tokens)


class _OllamaCompletions:
    def __init__(self, ollama_client: Any):
        self._client = ollama_client

    def create(self, model: str, messages: list, max_tokens: int = 4000,
               response_format: dict | None = None) -> _OllamaResponse:
        options = {"num_predict": max_tokens}
        format_param = "json" if response_format and response_format.get("type") == "json_object" else None

        response = self._client.chat(
            model=model,
            messages=messages,
            options=options,
            format=format_param,
        )
        content = response["message"]["content"]
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        logger.info(
            "Ollama call complete (input_tokens=%d, output_tokens=%d)",
            prompt_tokens, completion_tokens,
        )
        return _OllamaResponse(content, prompt_tokens, completion_tokens)


class _OllamaChat:
    def __init__(self, ollama_client: Any):
        self.completions = _OllamaCompletions(ollama_client)


class LLMClient:
    """Unified LLM client exposing chat.completions.create().
    Supports 'groq' and 'ollama' providers."""

    def __init__(self, provider: str = "groq", **kwargs):
        self.provider = provider

        if provider == "groq":
            import groq
            api_key = kwargs.get("api_key", "")
            self._backend = groq.Groq(api_key=api_key)
            self.chat = self._backend.chat
        elif provider == "ollama":
            import ollama
            base_url = kwargs.get("base_url", "http://localhost:11434")
            self._ollama = ollama.Client(host=base_url)
            self.chat = _OllamaChat(self._ollama)
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'groq' or 'ollama'.")


def create_llm_client(settings: "Settings") -> LLMClient:  # noqa: F821
    """Factory: creates the right LLM client based on settings."""
    llm_cfg = settings.llm
    if llm_cfg.provider == "groq":
        import os
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        return LLMClient(provider="groq", api_key=key)
    elif llm_cfg.provider == "ollama":
        return LLMClient(provider="ollama", base_url=llm_cfg.ollama_base_url)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")
