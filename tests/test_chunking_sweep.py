"""
The chunking sweep, offline: a hash embedder stands in for BGE and a recorder stands in
for MLflow. These tests check the plumbing and the relevance rule, not retrieval quality.
"""
import hashlib
import math
import sys
import types

import numpy as np

from eval import chunking_sweep as cs
from eval.ground_truth import load_questions
from src import providers
from src.corpus import chunk_corpus, load_corpus


class HashEmbedder:
    def encode(self, texts, normalize_embeddings=False):
        out = []
        for text in texts:
            v = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.stack(out)


def test_relevance_rule_reproduces_stored_labels_at_frozen_config():
    documents = load_corpus()
    chunks = chunk_corpus(documents, *cs.FROZEN)
    for q in load_questions():
        if q.answerable:
            assert cs.relevant_chunks(q, chunks) == set(q.relevant_chunk_ids), q.question_id


def test_sweep_never_reads_test_split(tmp_path):
    providers.set_embedder(HashEmbedder())
    report = cs.run(((800, 120),), tmp_path / "out.json", False, "unused")
    dev = [q for q in load_questions() if q.split == "dev" and q.answerable]
    assert report["stamp"]["split"] == "dev"
    assert report["configs"]["800/120"]["n"] == len(dev)
    assert report["configs"]["800/120"]["chunks"] == 314


def test_mlflow_gets_one_nested_run_per_config(tmp_path, monkeypatch):
    calls = []

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("mlflow")
    for name in ("set_experiment", "set_tags", "log_params", "log_param",
                 "log_artifact", "log_metrics"):
        setattr(fake, name, lambda *a, _n=name, **k: calls.append((_n, a, k)))
    fake.start_run = lambda **k: (calls.append(("start_run", (), k)), Run())[1]
    fake.get_tracking_uri = lambda: "fake://"
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    providers.set_embedder(HashEmbedder())
    cs.run(((600, 100), (800, 120)), tmp_path / "out.json", True, "exp")

    nested = [c for c in calls if c[0] == "start_run" and c[2].get("nested")]
    assert [c[2]["run_name"] for c in nested] == ["chunk-600/100", "chunk-800/120"]
    for name, args, _ in calls:
        if name == "log_metrics":
            assert not any(isinstance(v, float) and math.isnan(v) for v in args[0].values())


def test_parse_configs():
    assert cs._parse_configs("600/100, 800/120") == ((600, 100), (800, 120))
