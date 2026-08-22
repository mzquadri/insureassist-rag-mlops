"""
The committed reference run, and the query-side BGE prefix.

These assert the shape and internal consistency of the published artefact - not that the
numbers are good. The numbers are what they are; the tests exist so the file cannot drift
into claiming something the corpus and labels do not support.
"""
import json
from pathlib import Path

import pytest

from src import providers
from src.config import cfg
from src.corpus import chunk_corpus, config_hash, corpus_hash, load_corpus
from src.rag import retrieve

RUN_PATH = Path("eval/reference_run_nfip.json")


@pytest.fixture(scope="module")
def run():
    return json.loads(RUN_PATH.read_text(encoding="utf-8"))


class TestQueryPrefix:
    def test_the_query_is_prefixed_before_embedding(self, fake_store):
        """BGE is asymmetric: queries carry an instruction, passages do not."""
        seen = {}

        class RecordingEmbedder:
            def encode(self, text, normalize_embeddings=False):
                seen["text"] = text

                class V(list):
                    def tolist(self):
                        return list(self)

                return V([0.1] * 4)

        providers.set_embedder(RecordingEmbedder())
        providers.set_vector_store(fake_store)
        retrieve("Is a burst pipe covered?")

        assert seen["text"].startswith(cfg.BGE_QUERY_PREFIX)
        assert seen["text"].endswith("Is a burst pipe covered?")

    def test_prefix_is_the_documented_bge_instruction(self):
        assert cfg.BGE_QUERY_PREFIX == (
            "Represent this sentence for searching relevant passages: "
        )

    def test_prefix_can_be_disabled(self, monkeypatch, fake_store):
        seen = {}

        class RecordingEmbedder:
            def encode(self, text, normalize_embeddings=False):
                seen["text"] = text

                class V(list):
                    def tolist(self):
                        return list(self)

                return V([0.1] * 4)

        monkeypatch.setattr(cfg, "BGE_QUERY_PREFIX", "")
        providers.set_embedder(RecordingEmbedder())
        providers.set_vector_store(fake_store)
        retrieve("question")
        assert seen["text"] == "question"


class TestReferenceRunSchema:
    def test_top_level_sections(self, run):
        assert set(run) == {
            "run", "configuration", "corpus", "questions",
            "retrieval_metrics_answerable_only", "by_category", "by_relevant_form",
            "unanswerable", "latency_ms", "per_question",
        }

    def test_configuration_is_recorded(self, run):
        config = run["configuration"]
        assert config["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert config["embedding_dimension"] == 384
        assert config["chunk_size"] == 600
        assert config["chunk_overlap"] == 100
        assert config["distance"] == "cosine"
        assert config["query_prefix"]

    def test_config_hash_matches_the_recorded_configuration(self, run):
        config = run["configuration"]
        assert config["config_hash"] == config_hash(
            embedding_model=config["embedding_model"],
            size=config["chunk_size"],
            overlap=config["chunk_overlap"],
        )

    def test_corpus_hash_matches_the_committed_corpus(self, run):
        assert run["corpus"]["corpus_hash"] == corpus_hash(load_corpus())

    def test_corpus_counts_match_the_committed_corpus(self, run):
        documents = load_corpus()
        chunks = chunk_corpus(documents, 600, 100)
        assert run["corpus"]["documents"] == len(documents)
        assert run["corpus"]["chunks"] == len(chunks)

    def test_metrics_are_present_for_every_k(self, run):
        metrics = run["retrieval_metrics_answerable_only"]
        for k in (1, 3, 5):
            assert f"hit_rate@{k}" in metrics
            assert f"recall@{k}" in metrics
            assert f"precision@{k}" in metrics
        assert "mrr" in metrics

    def test_all_metrics_are_in_range(self, run):
        for name, value in run["retrieval_metrics_answerable_only"].items():
            assert 0.0 <= value <= 1.0, name

    def test_metrics_are_monotone_in_k(self, run):
        """Recall and hit rate cannot fall as k grows."""
        metrics = run["retrieval_metrics_answerable_only"]
        assert metrics["hit_rate@1"] <= metrics["hit_rate@3"] <= metrics["hit_rate@5"]
        assert metrics["recall@1"] <= metrics["recall@3"] <= metrics["recall@5"]


class TestUnanswerableAreSeparated:
    def test_unanswerable_are_excluded_from_the_aggregates(self, run):
        evaluated = [r for r in run["per_question"] if r["answerable"]]
        assert run["questions"]["unanswerable"] > 0
        # Aggregates are computed over answerable records only.
        assert all("metrics" in r for r in evaluated)
        assert all("metrics" not in r for r in run["per_question"] if not r["answerable"])

    def test_unanswerable_section_reports_no_recall(self, run):
        section = run["unanswerable"]
        assert section["count"] > 0
        assert "recall" not in section
        assert "abstention" not in json.dumps(section).lower().replace(
            "no abstention threshold is proposed", ""
        )

    def test_no_threshold_is_published(self, run):
        """No abstention threshold may appear until a threshold study is actually run."""
        assert "threshold_value" not in json.dumps(run)


class TestPerQuestionRecords:
    def test_every_question_has_a_record(self, run):
        assert len(run["per_question"]) == run["questions"]["evaluated"]

    def test_answerable_records_carry_their_labels(self, run):
        for record in run["per_question"]:
            if record["answerable"]:
                assert record["relevant_chunk_ids"]
                assert record["relevant_document_ids"]

    def test_retrieved_ids_are_real_chunk_ids(self, run):
        known = {c.chunk_id for c in chunk_corpus(load_corpus(), 600, 100)}
        for record in run["per_question"]:
            for chunk_id in record["retrieved_chunk_ids"]:
                assert chunk_id in known

    def test_latency_is_recorded(self, run):
        assert run["latency_ms"]["n"] == len(run["per_question"])
        assert run["latency_ms"]["p50"] > 0


class TestFailureAnalysis:
    def test_classifier_buckets_the_committed_run(self, run):
        from eval.failure_analysis import classify

        buckets = {classify(record) for record in run["per_question"]}
        assert buckets - {""}  # at least one classified outcome

    def test_a_top_one_hit_is_not_a_failure(self):
        from eval.failure_analysis import classify

        record = {
            "answerable": True, "category": "single_chunk",
            "relevant_chunk_ids": ["doc#aaa"], "retrieved_chunk_ids": ["doc#aaa"],
            "relevant_document_ids": ["doc"], "retrieved_document_ids": ["doc"],
        }
        assert classify(record) == ""

    def test_wrong_form_is_detected_from_the_top_hit(self):
        from eval.failure_analysis import classify

        record = {
            "answerable": True, "category": "near_miss",
            "relevant_chunk_ids": ["a#1"], "retrieved_chunk_ids": ["b#9", "a#2"],
            "relevant_document_ids": ["a"], "retrieved_document_ids": ["b", "a"],
        }
        assert classify(record) == "wrong near-duplicate form"

    def test_right_form_wrong_chunk_is_distinguished(self):
        from eval.failure_analysis import classify

        record = {
            "answerable": True, "category": "single_chunk",
            "relevant_chunk_ids": ["a#1"], "retrieved_chunk_ids": ["a#9"],
            "relevant_document_ids": ["a"], "retrieved_document_ids": ["a"],
        }
        assert classify(record) == "correct document, wrong chunk"
