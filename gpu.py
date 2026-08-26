"""Определение GPU-бэкенда (NVIDIA CUDA / AMD ROCm / CPU) и настройка
специфичных для него переменных окружения.

Важно: сборки PyTorch для ROCm используют тот же API-неймспейс torch.cuda,
что и обычный CUDA-torch (HIP мимикрирует CUDA) — поэтому
torch.cuda.is_available() возвращает True на обеих платформах, и движки
STT/TTS могут просто использовать device="cuda" без разветвления по вендору.
Разница только в переменных окружения и обходах багов конкретных сборок.
"""

import os


def configure_env() -> None:
    """Вызывать до первого import torch."""
    # Конфликт OpenMP-рантаймов между CTranslate2 (faster-whisper, CPU) и
    # GPU-torch — актуально для обоих вендоров, ставим всегда.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Ускоряет attention-ядра на ROCm; на CUDA-сборках переменная просто
    # игнорируется, вреда нет.
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def detect_backend() -> str:
    """Вызывать после import torch. Возвращает 'cuda', 'rocm' или 'cpu'."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm"
    return "cuda"


def backend_label(backend: str) -> str:
    import torch

    if backend == "rocm":
        return f"AMD ROCm (HIP {torch.version.hip})"
    if backend == "cuda":
        return f"NVIDIA CUDA {torch.version.cuda}"
    return "CPU"
