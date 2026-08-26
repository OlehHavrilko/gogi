"""Централизованная конфигурация.

Дефолты читаются из config.example.yaml (в репозитории), поверх них
накладывается config.yaml (в .gitignore, содержит пользовательские пути и
локальные переопределения). Если config.yaml нет — используются дефолты.
"""

import os
from pathlib import Path

import yaml

_ROOT = Path(__file__).parent
_DEFAULTS_FILE = _ROOT / "config.example.yaml"
_USER_FILE = _ROOT / "config.yaml"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


_config = _deep_merge(_load_yaml(_DEFAULTS_FILE), _load_yaml(_USER_FILE))

LLM_MODEL: str = _config["llm"]["model"]
SYSTEM_PROMPT: str = _config["llm"]["system_prompt"].strip()
MAX_TOOL_ITERATIONS: int = _config["llm"].get("max_tool_iterations", 4)

STT_MODEL_SIZE: str = _config["stt"]["model_size"]

TTS_NFE_STEP: int = _config["tts"]["nfe_step"]
TTS_CFG_STRENGTH: float = _config["tts"]["cfg_strength"]

# %APPDATA%-подобные переменные разворачиваются здесь, а не в config.yaml,
# поэтому один и тот же файл конфигурации работает для любого пользователя.
APPS: dict[str, str] = {
    name: os.path.expandvars(path) for name, path in _config.get("apps", {}).items()
}
ALIASES: dict[str, str] = dict(_config.get("aliases", {}))
