"""scripts/fetch_models.py: the image's fp16 model files (audit §9aac)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import fetch_models  # noqa: E402

torch = pytest.importorskip("torch")


def _tensors():
    return {"weight": torch.randn(4, 3), "position_ids": torch.arange(4)}


def test_safetensors_file_is_rewritten_in_fp16_and_ints_are_kept(tmp_path):
    from safetensors.torch import load_file, save_file

    path = str(tmp_path / "model.safetensors")
    original = _tensors()
    save_file(original, path)
    fetch_models._to_half(path)
    half = load_file(path)
    assert half["weight"].dtype == torch.float16
    assert half["position_ids"].dtype == torch.int64
    assert torch.equal(half["weight"], original["weight"].half())
    assert torch.equal(half["position_ids"], original["position_ids"])


def test_pytorch_bin_is_rewritten_in_fp16(tmp_path):
    path = str(tmp_path / "pytorch_model.bin")
    original = _tensors()
    torch.save(original, path)
    fetch_models._to_half(path)
    half = torch.load(path, weights_only=True)
    assert half["weight"].dtype == torch.float16
    assert half["position_ids"].dtype == torch.int64


def test_only_models_that_run_in_fp32_are_stored_in_fp16():
    # transformers 5 loads a checkpoint whose config names no dtype in the
    # checkpoint's own dtype: those models must stay out of _HALF.
    assert set(fetch_models._HALF) == {"BAAI/bge-large-en-v1.5", "urchade/gliner_multi"}
    assert set(fetch_models._HALF) <= set(fetch_models.MODELS)


def test_fetch_converts_only_when_asked(monkeypatch, tmp_path):
    converted = []
    monkeypatch.setattr(fetch_models, "_to_half", converted.append)
    monkeypatch.setattr(fetch_models, "MODELS", {"BAAI/bge-large-en-v1.5": "rev"})

    import huggingface_hub

    class _Api:
        def list_repo_files(self, repo, revision=None):
            return ["model.safetensors"]

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: str(tmp_path))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "diarization", type(sys)("diarization"))
    sys.modules["diarization"].fetch = lambda: {}
    monkeypatch.setitem(sys.modules, "rapidocr", None)

    fetch_models.fetch()
    assert converted == []
    fetch_models.fetch(half=True)
    assert converted == [os.path.join(str(tmp_path), "model.safetensors")]
