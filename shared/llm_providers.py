import json
import logging
import threading
import time
import urllib3
import requests
from typing import Optional, Dict, Any, Literal
from abc import ABC, abstractmethod
from . import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ProviderType = Literal["local", "cloud"]

LOCAL_PROVIDERS = {
    "Ollama": {"port": 11434, "endpoint_suffix": "/api/generate"},
    "LM Studio": {"port": 1234, "endpoint_suffix": "/v1/chat/completions"},
}

CLOUD_PROVIDERS = {
    "OpenAI": {"endpoint": "https://api.openai.com/v1/chat/completions"},
    "Azure OpenAI": {"endpoint": ""},
    "Anthropic": {"endpoint": "https://api.anthropic.com/v1/messages"},
}

TARGET_EMBEDDING_DIM = 1536


def normalize_embedding(emb: list[float], target_dim: int = TARGET_EMBEDDING_DIM) -> list[float]:
    if len(emb) < target_dim:
        return emb + [0.0] * (target_dim - len(emb))
    return emb[:target_dim]


logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message)
        self.provider = provider


class BaseLLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model = config.get("model", "")
        self._timeout = config.get("timeout", 120)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def test_connection(self) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        pass

    @property
    def model(self) -> str:
        return self._model

    @abstractmethod
    def get_provider_info(self) -> dict:
        pass


class LocalLLMProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._host = config.get("host", "localhost")
        self._port = config.get("port", 11434)
        self._base_url = f"http://{self._host}:{self._port}"
        provider_name = config.get("provider_name", "Ollama")
        from .llm_providers import LOCAL_PROVIDERS
        provider_config = LOCAL_PROVIDERS.get(provider_name, LOCAL_PROVIDERS["Ollama"])
        self._endpoint_suffix = provider_config.get("endpoint_suffix", "/api/generate")
        self._is_openai_compat = self._endpoint_suffix != "/api/generate"

    def _detect_backend(self) -> str:
        port_backends = {
            11434: "Ollama",
            1234: "LM Studio",
        }

        if self._port in port_backends:
            return port_backends[self._port]

        try:
            response = requests.get(f"{self._base_url}/", timeout=2)
            content = str(response.content).lower()
            if "ollama" in content:
                return "Ollama"
        except:
            pass

        return "Unknown Local LLM"

    def get_provider_info(self) -> dict:
        return {
            "type": "local",
            "backend": self._detect_backend(),
            "base_url": self._base_url,
            "host": self._host,
            "port": self._port,
            "model": self._model,
        }

    def get_available_models(self) -> list[str]:
        try:
            if self._endpoint_suffix == "/api/generate":
                response = requests.get(f"{self._base_url}/api/tags", timeout=5)
                response.raise_for_status()
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                return [m for m in models if m]
            else:
                response = requests.get(f"{self._base_url}/v1/models", timeout=5)
                response.raise_for_status()
                data = response.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                return [m for m in models if m]
        except Exception as e:
            raise LLMProviderError(f"Failed to fetch models: {e}", "local")

    @property
    def provider_type(self) -> ProviderType:
        return "local"

    def _pull_ollama_model(self, model: str):
        try:
            resp = requests.post(
                f"{self._base_url}/api/pull",
                json={"model": model},
                timeout=300,
            )
            resp.raise_for_status()
            logger.info(f"Auto-pulled Ollama model: {model}")
        except Exception as e:
            logger.warning(f"Failed to auto-pull Ollama model '{model}': {e}")

    def generate_embedding(self, text: str, embedding_model: Optional[str] = None) -> list[float]:
        from .config import EMBEDDING_MODEL as _cfg_embed_model
        model = embedding_model or _cfg_embed_model or "all-minilm"
        if self._is_openai_compat:
            payload = {"model": model, "input": text}
            url = f"{self._base_url}/v1/embeddings"
            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                emb = data["data"][0]["embedding"]
                return normalize_embedding(emb)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    raise LLMProviderError(
                        f"Embedding model '{model}' not found on LM Studio. "
                        f"Load an embedding model in LM Studio first.", "local")
                raise
        else:
            payload = {"model": model, "input": text}
            url = f"{self._base_url}/api/embed"
            try:
                resp = requests.post(url, json=payload, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                emb = data["embeddings"][0]
                return normalize_embedding(emb)
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    self._pull_ollama_model(model)
                    resp = requests.post(url, json=payload, timeout=self._timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    emb = data["embeddings"][0]
                    return normalize_embedding(emb)
                raise

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None, **kwargs) -> str:
        if self._is_openai_compat:
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
        else:
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens
            payload = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }

        try:
            response = requests.post(
                f"{self._base_url}{self._endpoint_suffix}",
                json=payload,
                timeout=self._timeout
            )
            response.raise_for_status()

            if self._is_openai_compat:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                return ""
            else:
                return response.json().get("response", "").strip()

        except requests.exceptions.ConnectionError:
            raise LLMProviderError(f"Cannot connect to {self._base_url}. Is the server running?", "local")
        except requests.exceptions.HTTPError as e:
            raise LLMProviderError(f"HTTP error: {e.response.status_code}", "local")
        except Exception as e:
            raise LLMProviderError(f"Error: {str(e)}", "local")

    def test_connection(self) -> tuple[bool, str]:
        try:
            if self._is_openai_compat:
                response = requests.post(
                    f"{self._base_url}{self._endpoint_suffix}",
                    json={"model": self._model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 10},
                    timeout=30
                )
            else:
                response = requests.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": "Say 'OK'", "stream": False},
                    timeout=30
                )
            response.raise_for_status()
            return True, f"Connected to {self._base_url}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to {self._base_url}"
        except Exception as e:
            return False, str(e)


class CloudLLMProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._provider_name = config.get("provider", "generic")
        self._api_key = config.get("api_key", "")
        self._endpoint = config.get("endpoint", "")
        self._verify = config.get("verify", True)
        self._aws_region = config.get("aws_region", "us-east-1")

    @property
    def provider_type(self) -> ProviderType:
        return "cloud"

    def get_provider_info(self) -> dict:
        if self._provider_name == "bedrock":
            return {
                "type": "cloud",
                "backend": "AWS Bedrock",
                "endpoint": self._get_bedrock_endpoint(),
                "model": self._model,
            }
        return {
            "type": "cloud",
            "backend": self._provider_name,
            "endpoint": self._endpoint,
            "model": self._model,
        }

    def get_available_models(self) -> list[str]:
        if self._provider_name == "bedrock":
            return []
        if not self._endpoint:
            return []
        base = self._endpoint.rstrip("/")
        if "/chat/completions" in base:
            base = base.replace("/chat/completions", "")
        models_url = f"{base}/models"
        try:
            resp = requests.get(models_url, headers=self._get_headers(),
                                timeout=self._timeout, verify=self._verify)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return []
            raise LLMProviderError(
                f"Failed to fetch models: {e.response.status_code}", "cloud")
        except Exception as e:
            raise LLMProviderError(f"Failed to fetch models: {e}", "cloud")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._provider_name == "openai":
            headers["Authorization"] = f"Bearer {self._api_key}"
        elif self._provider_name == "anthropic":
            headers["x-api-key"] = self._api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self._provider_name == "bedrock":
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def generate_embedding(self, text: str, embedding_model: Optional[str] = None) -> list[float]:
        from .config import EMBEDDING_MODEL as _cfg_embed_model
        model = embedding_model or _cfg_embed_model
        if self._provider_name == "bedrock":
            if not self._api_key:
                raise LLMProviderError("Bedrock API key not configured", "bedrock")
            embed_model = model or "amazon.titan-embed-text-v2:0"
            endpoint = (
                f"https://bedrock-runtime.{self._aws_region}.amazonaws.com/"
                f"model/{embed_model}/invoke"
            )
            payload = {"inputText": text}
            try:
                resp = requests.post(
                    endpoint, headers=self._get_headers(),
                    json=payload, timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding", [])
                if not emb:
                    raise LLMProviderError("Empty embedding from Bedrock", "bedrock")
                return normalize_embedding(emb)
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("message", "")
                except Exception:
                    detail = e.response.text[:200] if e.response.text else ""
                raise LLMProviderError(
                    f"Bedrock embedding error: {e.response.status_code} - {detail}", "bedrock")
        else:
            if not self._api_key:
                raise LLMProviderError("API key not configured", "cloud")
            embed_model = model or "text-embedding-3-small"
            embed_endpoint = self.config.get("embedding_endpoint", "").strip()
            if not embed_endpoint:
                base = self._endpoint.rstrip("/")
                if "/chat/completions" in base:
                    base = base.replace("/chat/completions", "")
                embed_endpoint = f"{base}/embeddings"
            payload = {"model": embed_model, "input": text}
            try:
                resp = requests.post(
                    embed_endpoint, headers=self._get_headers(),
                    json=payload, timeout=self._timeout, verify=self._verify,
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data["data"][0]["embedding"]
                return normalize_embedding(emb)
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    body = e.response.json()
                    detail = body.get("error", {}).get("message", "")
                except Exception:
                    detail = e.response.text[:200] if e.response.text else ""
                raise LLMProviderError(
                    f"Embedding API error: {e.response.status_code} - {detail}", "cloud")

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None, **kwargs) -> str:
        if self._provider_name == "bedrock":
            return self._generate_bedrock(prompt, temperature, max_tokens)

        if not self._api_key:
            raise LLMProviderError("API key not configured", "cloud")
        if not self._endpoint:
            raise LLMProviderError("API endpoint not configured", "cloud")

        if self._provider_name == "openai":
            return self._generate_openai(prompt, temperature, max_tokens)
        else:
            raise LLMProviderError(f"Provider {self._provider_name} not supported", "cloud")

    def _generate_openai(self, prompt: str, temperature: float, max_tokens: Optional[int]) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                self._endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self._timeout,
                verify=self._verify
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            return ""
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                body = e.response.json()
                detail = body.get("error", {}).get("message", "")
            except Exception:
                detail = e.response.text[:200] if e.response.text else ""
            msg = f"API error: {e.response.status_code}"
            if detail:
                msg += f" - {detail}"
            raise LLMProviderError(msg, "openai")
        except Exception as e:
            raise LLMProviderError(f"Error: {str(e)}", "cloud")

    def _get_bedrock_endpoint(self) -> str:
        return f"https://bedrock-runtime.{self._aws_region}.amazonaws.com/model/{self._model}/converse"

    def _generate_bedrock(self, prompt: str, temperature: float = 0.3, max_tokens: Optional[int] = None) -> str:
        if not self._api_key:
            raise LLMProviderError("Bedrock API key not configured", "bedrock")
        if not self._model:
            raise LLMProviderError("Model not configured", "bedrock")

        payload = {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens or 4000, "temperature": temperature},
        }
        endpoint = self._get_bedrock_endpoint()

        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            text_blocks = data.get("output", {}).get("message", {}).get("content", [])
            if text_blocks:
                return text_blocks[0].get("text", "").strip()
            return ""
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                body_data = e.response.json()
                detail = body_data.get("message", "")
            except Exception:
                detail = e.response.text[:200] if e.response.text else ""
            msg = f"API error: {e.response.status_code}"
            if detail:
                msg += f" - {detail}"
            raise LLMProviderError(msg, "bedrock")
        except Exception as e:
            raise LLMProviderError(f"Error: {str(e)}", "bedrock")

    def test_connection(self) -> tuple[bool, str]:
        if self._provider_name == "bedrock":
            if not self._api_key:
                return False, "Bedrock API key not configured"
            try:
                result = self.generate("Say 'OK'", max_tokens=10)
                return True, f"Connected to AWS Bedrock ({self._aws_region})"
            except Exception as e:
                return False, str(e)

        if not self._api_key:
            return False, "API key not configured"
        try:
            result = self.generate("Say 'OK'", max_tokens=10)
            return True, f"Connected to {self._provider_name}"
        except Exception as e:
            return False, str(e)


class LLMClient:
    _provider: Optional[BaseLLMProvider] = None
    _lock = threading.Lock()

    @classmethod
    def set_provider(cls, provider: BaseLLMProvider):
        with cls._lock:
            cls._provider = provider
            logger.info(f"LLM provider set to: {provider.provider_type}")

    @classmethod
    def get_provider(cls) -> Optional[BaseLLMProvider]:
        with cls._lock:
            return cls._provider

    @classmethod
    def get_provider_info(cls) -> Optional[dict]:
        with cls._lock:
            if cls._provider is None:
                return None
            return cls._provider.get_provider_info()

    @classmethod
    def generate(cls, prompt: str, **kwargs) -> str:
        with cls._lock:
            if cls._provider is None:
                raise LLMProviderError("No provider configured", "unknown")

            from .config import _config
            expected_type = _config.get("llm_provider_type", "local")
            if cls._provider.provider_type != expected_type:
                raise LLMProviderError(
                    f"Provider type mismatch: active is '{cls._provider.provider_type}', "
                    f"config says '{expected_type}'",
                    cls._provider.provider_type
                )

            logger.info(f"Generating with provider type: {cls._provider.provider_type}, model: {cls._provider.model}")

            if config.PROMPT_LOGGING_ENABLED:
                try:
                    with open(config.PROMPT_LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"[{datetime.now().isoformat()}] PROMPT SENT TO LLM\n")
                        f.write(f"{'='*60}\n")
                        f.write(prompt)
                        f.write(f"\n{'='*60}\n\n")
                except Exception:
                    logger.exception("Failed to write prompt log")

            return cls._provider.generate(prompt, **kwargs)

    @classmethod
    def generate_embedding(cls, text: str, embedding_model: Optional[str] = None) -> list[float]:
        with cls._lock:
            if cls._provider is None:
                raise LLMProviderError("No provider configured", "unknown")
            from .config import _config
            expected_type = _config.get("llm_provider_type", "local")
            if cls._provider.provider_type != expected_type:
                raise LLMProviderError(
                    f"Provider type mismatch: active is '{cls._provider.provider_type}', "
                    f"config says '{expected_type}'",
                    cls._provider.provider_type
                )
            return cls._provider.generate_embedding(text, embedding_model)

    @classmethod
    def test_connection(cls) -> tuple[bool, str]:
        with cls._lock:
            if cls._provider is None:
                return False, "No provider configured"
            return cls._provider.test_connection()
