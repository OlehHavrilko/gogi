"""Проверка определения GPU-бэкенда без реального железа — мокаем torch."""

import sys
import types

import gpu


def _fake_torch(cuda_available: bool, hip_version: str | None, cuda_version: str | None):
    mod = types.SimpleNamespace()
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    mod.version = types.SimpleNamespace(hip=hip_version, cuda=cuda_version)
    return mod


def test_detect_backend_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, None, None))
    assert gpu.detect_backend() == "cpu"


def test_detect_backend_nvidia_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.4"))
    assert gpu.detect_backend() == "cuda"


def test_detect_backend_amd_rocm(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "7.2.1", None))
    assert gpu.detect_backend() == "rocm"


def test_backend_label_formats(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, "7.2.1", None))
    assert "ROCm" in gpu.backend_label("rocm")

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True, None, "12.4"))
    assert "CUDA" in gpu.backend_label("cuda")

    assert gpu.backend_label("cpu") == "CPU"


def test_configure_env_sets_expected_vars(monkeypatch):
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", raising=False)
    gpu.configure_env()
    import os

    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] == "1"
