"""Проверка загрузки и слияния конфигурации."""

import os

import config


def test_defaults_loaded():
    assert config.LLM_MODEL
    assert config.SYSTEM_PROMPT
    assert config.MAX_TOOL_ITERATIONS >= 1
    assert config.STT_MODEL_SIZE
    assert isinstance(config.TTS_NFE_STEP, int)
    assert isinstance(config.TTS_CFG_STRENGTH, float)


def test_apps_and_aliases_present():
    assert "chrome" in config.APPS
    assert "хром" in config.ALIASES
    assert config.ALIASES["хром"] == "chrome"


def test_deep_merge_overrides_nested_keys_only():
    base = {"llm": {"model": "a", "max_tool_iterations": 4}, "apps": {"chrome": "x"}}
    override = {"llm": {"model": "b"}, "apps": {"notepad": "y"}}
    merged = config._deep_merge(base, override)

    assert merged["llm"]["model"] == "b"
    assert merged["llm"]["max_tool_iterations"] == 4  # не тронуто
    assert merged["apps"] == {"chrome": "x", "notepad": "y"}  # слито, не заменено


def test_app_paths_expand_env_vars():
    # vscode задан через %LOCALAPPDATA% в config.example.yaml — после загрузки
    # конфига переменная должна быть развёрнута, а не оставаться литералом.
    assert "%LOCALAPPDATA%" not in config.APPS["vscode"]
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    assert not local_appdata or local_appdata in config.APPS["vscode"]
