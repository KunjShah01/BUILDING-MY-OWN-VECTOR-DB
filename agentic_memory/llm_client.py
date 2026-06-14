import json
import os
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any


class LLMClient:
    """Zero-dep LLM client supporting Ollama and OpenAI-compatible APIs.

    Uses stdlib urllib only. No openai SDK, no httpx, no requests.
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "",
        base_url: str = "",
        api_key: str = "",
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._resolve_defaults()

    def _resolve_defaults(self):
        if self.provider == "ollama":
            self.base_url = self.base_url or os.environ.get(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            self.model = self.model or os.environ.get("OLLAMA_LLM_MODEL", "llama3.1")
            self.api_key = ""  # Ollama doesn't need auth
        elif self.provider == "openai":
            self.base_url = self.base_url or "https://api.openai.com/v1"
            self.model = self.model or os.environ.get(
                "OPENAI_LLM_MODEL", "gpt-4o-mini"
            )
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_chat_url(self) -> str:
        if self.provider == "ollama":
            return f"{self.base_url}/v1/chat/completions"
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            self._get_chat_url(),
            data=payload,
            headers=self._get_headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
                choices = body.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            print(f"LLM API error {e.code}: {error_body}")
        except Exception as e:
            print(f"LLM request failed: {e}")
        return None

    def embed(self, text: str) -> Optional[List[float]]:
        if self.provider == "ollama":
            url = f"{self.base_url}/api/embed"
            payload = json.dumps({
                "model": os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                "input": text,
            }).encode()
        else:
            url = f"{self.base_url}/embeddings"
            payload = json.dumps({
                "model": self.model,
                "input": text,
            }).encode()

        req = urllib.request.Request(
            url, data=payload, headers=self._get_headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read())
                if self.provider == "ollama":
                    embeddings = body.get("embeddings", [])
                    return embeddings[0] if embeddings else None
                else:
                    data = body.get("data", [])
                    return data[0].get("embedding") if data else None
        except Exception as e:
            print(f"Embedding API error: {e}")
            return None
