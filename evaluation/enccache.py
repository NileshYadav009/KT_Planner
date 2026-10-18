"""Disk memo for model outputs, for experiments only (never in the app).

import enccache; enccache.install(path) BEFORE any model is constructed.
Patches SentenceTransformer.encode, CrossEncoder.predict and
GLiNER.predict_entities so the same input (same model) is computed once
across runs. Embeddings are computed per uncached text in one batch and
stored as float32.
"""
import hashlib
import json
import pickle
import os
import sqlite3
import threading

import numpy as np

_db = None
_lock = threading.Lock()
STATS = {"hit": 0, "miss": 0}


def _key(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _get_many(keys):
    out = {}
    with _lock:
        cur = _db.cursor()
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            q = "SELECT k, v FROM memo WHERE k IN (%s)" % ",".join("?" * len(chunk))
            for k, v in cur.execute(q, chunk):
                out[k] = pickle.loads(v)
    return out


def _put_many(items):
    with _lock:
        _db.executemany("INSERT OR REPLACE INTO memo (k, v) VALUES (?, ?)",
                        [(k, pickle.dumps(v, protocol=4)) for k, v in items])
        _db.commit()


def install(path):
    global _db
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _db = sqlite3.connect(path, check_same_thread=False)
    _db.execute("CREATE TABLE IF NOT EXISTS memo (k TEXT PRIMARY KEY, v BLOB)")
    _db.commit()

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    st_init = SentenceTransformer.__init__

    def st_init_tagged(self, *a, **k):
        st_init(self, *a, **k)
        self._memo_tag = str(a[0] if a else k.get("model_name_or_path"))

    SentenceTransformer.__init__ = st_init_tagged
    st_encode = SentenceTransformer.encode

    def encode(self, sentences, *a, **k):
        tag = getattr(self, "_memo_tag", "?")
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        norm = bool(k.get("normalize_embeddings", False))
        keys = [_key("st", tag, norm, t) for t in texts]
        found = _get_many(list(set(keys)))
        todo = [i for i, kk in enumerate(keys) if kk not in found]
        STATS["hit"] += len(texts) - len(todo)
        STATS["miss"] += len(todo)
        if todo:
            uniq = list(dict.fromkeys(texts[i] for i in todo))
            kw = dict(k)
            kw["convert_to_tensor"] = False
            kw["convert_to_numpy"] = True
            vecs = st_encode(self, uniq, **kw)
            new = {_key("st", tag, norm, t): np.asarray(v, dtype=np.float32) for t, v in zip(uniq, vecs)}
            _put_many(list(new.items()))
            found.update(new)
        arr = np.stack([found[kk] for kk in keys]) if texts else np.zeros((0, 1), dtype=np.float32)
        if single:
            arr = arr[0]
        if k.get("convert_to_tensor"):
            return torch.from_numpy(np.ascontiguousarray(arr))
        return arr

    SentenceTransformer.encode = encode

    ce_init = CrossEncoder.__init__

    def ce_init_tagged(self, *a, **k):
        ce_init(self, *a, **k)
        self._memo_tag = str(a[0] if a else k.get("model_name_or_path") or k.get("model_name"))

    CrossEncoder.__init__ = ce_init_tagged
    ce_predict = CrossEncoder.predict

    def predict(self, sentences, *a, **k):
        tag = getattr(self, "_memo_tag", "?")
        single = isinstance(sentences, tuple) or (isinstance(sentences, list) and sentences and isinstance(sentences[0], str))
        pairs = [tuple(sentences)] if single else [tuple(p) for p in sentences]
        extra = json.dumps(sorted((kk, str(v)) for kk, v in k.items()))
        keys = [_key("ce", tag, extra, p[0], p[1]) for p in pairs]
        found = _get_many(list(set(keys)))
        todo = [i for i, kk in enumerate(keys) if kk not in found]
        STATS["hit"] += len(pairs) - len(todo)
        STATS["miss"] += len(todo)
        if todo:
            uniq = list(dict.fromkeys(pairs[i] for i in todo))
            scores = ce_predict(self, [list(p) for p in uniq], *a, **k)
            new = {_key("ce", tag, extra, p[0], p[1]): np.asarray(s) for p, s in zip(uniq, scores)}
            _put_many(list(new.items()))
            found.update(new)
        arr = np.array([found[kk] for kk in keys])
        return arr[0] if single else arr

    CrossEncoder.predict = predict

    # GLiNER.from_pretrained returns a subclass instance: wrap the instance
    # method once the app's EntityExtractor has loaded it.
    import entity_extractor
    ee_init = entity_extractor.EntityExtractor.__init__

    def ee_init_memo(self, *a, **k):
        ee_init(self, *a, **k)
        model = getattr(self, "model", None)
        if model is None or getattr(model, "_memo_wrapped", False):
            return
        g_predict = model.predict_entities

        def predict_entities(text, labels, *pa, **pk):
            kk = _key("gl", json.dumps(list(labels)), json.dumps([str(x) for x in pa]),
                      json.dumps(sorted((x, str(v)) for x, v in pk.items())), text)
            found = _get_many([kk])
            if kk in found:
                STATS["hit"] += 1
                return found[kk]
            STATS["miss"] += 1
            res = g_predict(text, labels, *pa, **pk)
            _put_many([(kk, res)])
            return res

        model.predict_entities = predict_entities
        model._memo_wrapped = True

    entity_extractor.EntityExtractor.__init__ = ee_init_memo
