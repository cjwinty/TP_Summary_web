import os
import json
import base64
import logging
import threading

def get_base_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(get_base_dir(), ".env"))
except ImportError:
    pass

log_file = os.path.join(get_base_dir(), "tp_query_error.log")
logging.basicConfig(
    level=logging.INFO,
    filename=log_file,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("TP_BASE_URL", "https://damarel.tpondemand.com/api/v2")
USERNAME = os.getenv("TP_USERNAME")
PASSWORD = os.getenv("TP_PASSWORD")
PROJECT_NAME = os.getenv("TP_PROJECT_NAME", "External Support")
VERSION = "1.4.1"
TP_API_TOKEN = os.getenv("TP_API_TOKEN")

CONFIG_FILE = os.path.join(get_base_dir(), "user_config.json")
SECURE_CONFIG_FILE = os.path.join(get_base_dir(), "secure_config.json")


def load_user_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_user_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def _encode(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def _decode(value: str) -> str:
    return base64.b64decode(value.encode()).decode()


def load_secure_config() -> dict:
    try:
        with open(SECURE_CONFIG_FILE, "r") as f:
            enc_data = json.load(f)
        return {k: _decode(v) for k, v in enc_data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_secure_config(config: dict):
    enc_data = {k: _encode(v) for k, v in config.items()}
    with open(SECURE_CONFIG_FILE, "w") as f:
        json.dump(enc_data, f, indent=2)


_config = load_user_config()
_secure_config = load_secure_config()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = _config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3.2")

EMBEDDING_MODEL = _config.get("embedding_model", "")

LLM_PROVIDER_TYPE = _config.get("llm_provider_type", "local")
LOCAL_PROVIDER = _config.get("llm_local_provider", "Ollama")
LOCAL_LLM_HOST = _config.get("llm_local_host", "localhost")

CLOUD_CONFIG = {
    "provider": _config.get("llm_cloud_provider", "openai"),
    "endpoint": _config.get("llm_cloud_endpoint", "https://api.openai.com/v1/chat/completions"),
    "api_key": _secure_config.get("llm_api_key", ""),
    "aws_region": _config.get("llm_aws_region", "us-east-1"),
    "model": _config.get("llm_cloud_model", "gpt-4"),
}

PROMPT_LOG_FILE = os.path.join(get_base_dir(), "tp_prompt_log.txt")
PROMPT_LOGGING_ENABLED = _config.get("prompt_logging_enabled", False)

VERIFY_SSL = _config.get("verify_ssl", True)


def set_ollama_model(model_name):
    global OLLAMA_MODEL
    _config["ollama_model"] = model_name
    OLLAMA_MODEL = model_name
    save_user_config(_config)


def set_embedding_model(model_name: str):
    global EMBEDDING_MODEL
    _config["embedding_model"] = model_name
    EMBEDDING_MODEL = model_name
    save_user_config(_config)


def set_llm_provider_type(provider_type: str):
    global LLM_PROVIDER_TYPE
    _config["llm_provider_type"] = provider_type
    LLM_PROVIDER_TYPE = provider_type
    save_user_config(_config)


def set_local_provider(provider_name: str):
    global LOCAL_PROVIDER
    _config["llm_local_provider"] = provider_name
    LOCAL_PROVIDER = provider_name
    save_user_config(_config)


def set_local_host(host: str):
    global LOCAL_LLM_HOST
    _config["llm_local_host"] = host
    LOCAL_LLM_HOST = host
    save_user_config(_config)


def set_cloud_config(provider: str, endpoint: str, api_key: str, model: str, embedding_endpoint: str = ""):
    _config["llm_cloud_provider"] = provider
    _config["llm_cloud_endpoint"] = endpoint
    _config["llm_cloud_model"] = model
    _config["llm_embedding_endpoint"] = embedding_endpoint
    save_user_config(_config)
    if api_key is not None:
        _secure_config["llm_api_key"] = api_key
        save_secure_config(_secure_config)
    CLOUD_CONFIG.update({
        "provider": provider,
        "endpoint": endpoint,
        "api_key": _secure_config.get("llm_api_key", ""),
        "model": model,
        "aws_region": _config.get("llm_aws_region", "us-east-1"),
        "embedding_endpoint": embedding_endpoint,
    })


def set_bedrock_config(region: str, api_key: str, model: str):
    _config["llm_cloud_provider"] = "bedrock"
    _config["llm_aws_region"] = region
    _config["llm_cloud_model"] = model
    save_user_config(_config)
    if api_key is not None:
        _secure_config["llm_bedrock_api_key"] = api_key
        save_secure_config(_secure_config)
    CLOUD_CONFIG.update({
        "provider": "bedrock",
        "aws_region": region,
        "model": model,
        "endpoint": "",
        "api_key": _secure_config.get("llm_bedrock_api_key", ""),
    })


def set_prompt_logging_enabled(enabled: bool):
    global PROMPT_LOGGING_ENABLED
    PROMPT_LOGGING_ENABLED = enabled
    _config["prompt_logging_enabled"] = enabled
    save_user_config(_config)


def set_verify_ssl(enabled: bool):
    global VERIFY_SSL
    VERIFY_SSL = enabled
    _config["verify_ssl"] = enabled
    save_user_config(_config)


def initialize_llm():
    from .llm_providers import LLMClient, LocalLLMProvider, CloudLLMProvider, LLMProviderError, LOCAL_PROVIDERS

    llm_provider_type = _config.get("llm_provider_type", "local")

    with _llm_lock:
        if llm_provider_type == "cloud":
            provider_name = _config.get("llm_cloud_provider", "openai")
            if provider_name == "bedrock":
                cloud_config = {
                    "provider": "bedrock",
                    "api_key": _secure_config.get("llm_bedrock_api_key", ""),
                    "aws_region": _config.get("llm_aws_region", "us-east-1"),
                    "model": _config.get("llm_cloud_model", "gpt-4"),
                }
                if not cloud_config.get("api_key"):
                    raise LLMProviderError("Bedrock API key not configured", "cloud")
                provider = CloudLLMProvider(cloud_config)
                logger.info("Using cloud LLM: AWS Bedrock")
            else:
                cloud_config = {
                    "provider": provider_name,
                    "endpoint": _config.get("llm_cloud_endpoint", "https://api.openai.com/v1/chat/completions"),
                    "api_key": _secure_config.get("llm_api_key", ""),
                    "model": _config.get("llm_cloud_model", "gpt-4"),
                    "verify": VERIFY_SSL,
                    "embedding_endpoint": _config.get("llm_embedding_endpoint", ""),
                }
                if not cloud_config.get("api_key"):
                    raise LLMProviderError("Cloud LLM selected but API key not configured", "cloud")
                provider = CloudLLMProvider(cloud_config)
                logger.info(f"Using cloud LLM: {provider_name}")
        else:
            ollama_model = _config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3.2")
            local_provider = _config.get("llm_local_provider", "Ollama")

            if not ollama_model:
                raise LLMProviderError("Local LLM selected but model not configured", "local")
            local_provider_config = LOCAL_PROVIDERS.get(local_provider, LOCAL_PROVIDERS["Ollama"])
            local_config = {
                "host": _config.get("llm_local_host", "localhost"),
                "port": local_provider_config["port"],
                "model": ollama_model,
                "timeout": 120,
                "provider_name": local_provider,
            }
            provider = LocalLLMProvider(local_config)
            logger.info(f"Using local LLM: {local_provider} with model {ollama_model}")

        LLMClient.set_provider(provider)

    return LLMClient


def validate_env():
    errors = []
    warnings = []
    if not TP_API_TOKEN and not USERNAME and not PASSWORD:
        errors.append("No API credentials configured — set TP_API_TOKEN or TP_USERNAME/TP_PASSWORD in .env")
    elif TP_API_TOKEN:
        warnings.append("TP_API_TOKEN set — using Bearer token authentication")
    else:
        warnings.append("TP_API_TOKEN not set, falling back to basic auth (TP_USERNAME/TP_PASSWORD)")
    if not BASE_URL:
        errors.append("TP_BASE_URL not set — add it to .env")
    for w in warnings:
        logger.info("validate_env: %s", w)
    return len(errors) == 0, errors


def validate_llm_config():
    llm_provider_type = _config.get("llm_provider_type", "local")

    if llm_provider_type == "cloud":
        errors = []
        provider_name = _config.get("llm_cloud_provider", "openai")
        if provider_name == "bedrock":
            if not _secure_config.get("llm_bedrock_api_key"):
                errors.append("Bedrock API key not configured")
            if not _config.get("llm_aws_region"):
                errors.append("AWS region not configured")
        else:
            if not _secure_config.get("llm_api_key"):
                errors.append("API key not configured")
        if not _config.get("llm_cloud_model"):
            errors.append("Model not configured")
        return len(errors) == 0, errors
    else:
        errors = []
        ollama_model = _config.get("ollama_model") or os.getenv("OLLAMA_MODEL", "llama3.2")
        if not ollama_model:
            errors.append("Model not configured")
        return len(errors) == 0, errors


_llm_lock = threading.Lock()
