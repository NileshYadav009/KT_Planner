"""Each text embedded once per model in a process (audit P1-5, P1-6).

Field population encoded a section's sentences again for every field of
the section, though classification had just encoded the same sentences
with the same model: on a 2,400-word KT that took longer than everything
else together. Embeddings are now kept by text, for the model and the
normalisation asked for, most recently used first, up to MAX_TEXTS per
process (about 4 KB each for bge-large).

Texts not seen before are encoded in one batch and kept as float32, as
evaluation/enccache.py has always done for the accuracy sets, so the
placements measured there and the ones made here start from the same
vectors.
"""
import itertools
import threading
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np

MAX_TEXTS = 10_000

_lock = threading.Lock()
_memo: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_tokens = itertools.count(1)


def _token(model) -> Optional[int]:
    # A token of the model's own, not id(): a model freed and another made
    # at the same address would otherwise be served the first one's vectors.
    token = getattr(model, "_embedding_memo_token", None)
    if token is None:
        with _lock:
            token = getattr(model, "_embedding_memo_token", None) or next(_tokens)
        try:
            setattr(model, "_embedding_memo_token", token)
        except Exception:
            return None
    return token


def encode(model, texts: Union[str, Sequence[str]], normalize_embeddings: bool = True,
           convert_to_tensor: bool = True):
    """model.encode(texts, ...) from memory where it can be: a 1-D result for
    one text, 2-D for a list, a CPU torch tensor if `convert_to_tensor`."""
    single = isinstance(texts, str)
    items: List[str] = [texts] if single else list(texts)
    token = _token(model)
    keys = [(token, bool(normalize_embeddings), t) for t in items]
    with _lock:
        found = {k: _memo[k] for k in keys if token is not None and k in _memo}
        for k in found:
            _memo.move_to_end(k)
    todo = list(dict.fromkeys(k for k in keys if k not in found))
    if todo:
        try:
            vectors = model.encode([k[2] for k in todo], convert_to_tensor=False, convert_to_numpy=True,
                                   normalize_embeddings=normalize_embeddings)
            new = {k: np.asarray(v, dtype=np.float32) for k, v in zip(todo, vectors)}
        except (TypeError, ValueError):
            # Not a sentence-transformers model (a test's stand-in): asked
            # as it always was, and nothing kept.
            return model.encode(texts, convert_to_tensor=convert_to_tensor, normalize_embeddings=normalize_embeddings)
        found.update(new)
        if token is not None:
            with _lock:
                _memo.update(new)
                while len(_memo) > MAX_TEXTS:
                    _memo.popitem(last=False)
    if not items:
        array = np.zeros((0, 1), dtype=np.float32)
    else:
        array = np.stack([found[k] for k in keys])
    if single:
        array = array[0]
    if convert_to_tensor:
        import torch
        return torch.from_numpy(np.ascontiguousarray(array))
    return array


def clear() -> None:
    with _lock:
        _memo.clear()
