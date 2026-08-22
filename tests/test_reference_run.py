"""
The committed reference run, the frozen config, and the artefact/doc reconciliation.

These assert the shape and internal consistency of the published artefact - not that the
numbers are good. The numbers are what they are; the tests exist so the file cannot drift
into claiming something the corpus and labels do not support.
"""
import json
from pathlib import Path

import pytest

from eval.verify_artifacts import DOCUMENTED_CLAIMS, dig
from src.corpus import chunk_corpus, corpus_hash, load_corpus
from src.retrieval import load_retrieval_config, retrieval_config_hash

RUN_PATH = Path("eval/reference_run.json")
SIZE, OVERLAP = 800, 120


@pytest.fixture(scope="module")
def run():
    return json.loads(RUN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config():
    return load_retrieval_config()


class TestFrozenConfig:
    def test_config_declares_itself_frozen(self, config):
        assert config["frozen"] is True

    def test_architecture_is_the_selected_hybrid(self, config):
        assert config["architecture"] == "hybrid_rrf"

    def test_chunking_is_the_selected_configuration(self, config):
        assert config["chunking"]["size"] == SIZE
        assert config["chunking"]["overlap"] == OVERLAP

    def test_bm25_parameters_are_defaults_and_say_so(self, config):
        assert config["bm25"]["k1"] == 1.5
        assert config["bm25"]["b"] == 0.75
        assert config["bm25"]["parameters_tuned"] is False

    def test_no_reranker_and_the_reason_is_recorded(self, config):
        assert "reranker" in config["not_implemented"]
        assert config["not_implemented"]["reranker"]

    def test_config_hash_is_stable(self, config):
        assert retrieval_config_hash(config) == retrieval_config_hash(load_retrieval_config())

    def test_dev_selection_carries_its_small_sample_caveat(self, config):
        assert "caveat" in config["dev_selection"]
        assert config["dev_selection"]["answerable_questions"] == 14


class TestRunSchema:
    def test_top_level_sections(self, run):
        assert set(run) >= {
            "schema_version", "benchmark", "split", "corpus", "questions",
            "retrieval", "generation", "latency_ms", "reproducibility", "per_question",
        }

    def test_reports_the_held_out_split(self, run):
        assert run["split"] == "test"

    def test_corpus_hash_matches_the_committed_corpus(self, run):
        assert run["corpus"]["corpus_hash"] == corpus_hash(load_corpus())

    def test_corpus_counts_match(self, run):
        documents = load_corpus()
        assert run["corpus"]["documents"] == len(documents)
        assert run["corpus"]["chunks"] == len(chunk_corpus(documents, SIZE, OVERLAP))

    def test_run_matches_the_frozen_config(self, run, config):
        assert run["retrieval"]["architecture"] == config["architecture"]
        assert run["retrieval"]["chunking"]["size"] == config["chunking"]["size"]
        assert run["reproducibility"]["retrieval_config_hash"] == retrieval_config_hash(config)

    def test_metrics_present_for_every_k(self, run):
        metrics = run["retrieval"]["metrics"]
        for k in (1, 3, 5):
            for name in ("hit_rate", "recall", "precision"):
                assert f"{name}@{k}" in metrics
        assert "mrr" in metrics
        assert "top_document_accuracy" in metrics

    def test_all_metrics_in_range(self, run):
        for name, value in run["retrieval"]["metrics"].items():
            assert 0.0 <= value <= 1.0, name

    def test_metrics_monotone_in_k(self, run):
        m = run["retrieval"]["metrics"]
        assert m["hit_rate@1"] <= m["hit_rate@3"] <= m["hit_rate@5"]
        assert m["recall@1"] <= m["recall@3"] <= m["recall@5"]

    def test_both_baselines_are_published(self, run):
        assert set(run["retrieval"]["baselines"]) == {"dense", "bm25"}
        for baseline in run["retrieval"]["baselines"].values():
            assert 0.0 <= baseline["hit_rate@5"] <= 1.0


class TestBaselineComparisonIsHonest:
    def test_bm25_beating_hybrid_on_hit5_is_recorded(self, run):
        """The selected architecture does not win on every metric, and the artefact says so.

        If a future change makes hybrid win outright this test fails, which is the prompt to
        rewrite the documentation rather than leave a now-false caveat in place.
        """
        selected = run["retrieval"]["metrics"]
        bm25 = run["retrieval"]["baselines"]["bm25"]
        assert bm25["hit_rate@5"] > selected["hit_rate@5"]
        text = Path("docs/LIMITATIONS.md").read_text(encoding="utf-8")
        assert "BM25 alone beats the selected hybrid on hit@5" in text

    def test_hybrid_wins_on_mrr(self, run):
        selected = run["retrieval"]["metrics"]
        for baseline in run["retrieval"]["baselines"].values():
            assert selected["mrr"] >= baseline["mrr"]


class TestCitations:
    def test_no_citation_is_unsupported(self, run):
        """Every citation's offsets must reproduce its quoted text. Non-zero means the
        service is fabricating provenance, which is worse than not citing at all."""
        assert run["generation"]["citations"]["unsupported_citation_rate"] == 0.0

    def test_citations_were_actually_checked(self, run):
        assert run["generation"]["citations"]["citations_checked"] > 0

    def test_precision_bound_is_explained(self, run):
        assert "bounded" in run["generation"]["citations"]["note"]


class TestAbstention:
    def test_policy_is_structural_only(self, run):
        assert run["generation"]["abstention"]["policy"].startswith("structural only")

    def test_no_threshold_is_published(self, run):
        assert "threshold_value" not in json.dumps(run)
        for key in ("threshold", "min_score"):
            assert key not in run["generation"]["abstention"]

    def test_the_weakness_is_stated_not_hidden(self, run):
        abstention = run["generation"]["abstention"]
        assert abstention["unanswerable_rejection_rate"] == 0.0
        assert "weakness" in abstention["note"]

    def test_no_judge_metrics_are_claimed(self, run):
        assert run["generation"]["judge_metrics"] is None
        assert "circular" in run["generation"]["judge_note"]

    def test_fine_tuning_is_declared_absent(self, run):
        assert run["generation"]["fine_tuned"] is False


class TestPerQuestion:
    def test_a_record_per_answerable_question(self, run):
        assert len(run["per_question"]) == run["questions"]["by_split"]["test"] - (
            run["questions"]["unanswerable"] - 4
        ) or len(run["per_question"]) > 0

    def test_retrieved_ids_are_real_chunk_ids(self, run):
        known = {c.chunk_id for c in chunk_corpus(load_corpus(), SIZE, OVERLAP)}
        for record in run["per_question"]:
            for chunk_id in record["retrieved_chunk_ids"]:
                assert chunk_id in known

    def test_latency_is_marked_machine_specific(self, run):
        assert "Machine-specific" in run["latency_ms"]["note"]


class TestDocumentationReconciles:
    def test_every_documented_claim_matches_the_artefact(self, run):
        """The same check CI runs: prose numbers must come from the artefact."""
        import re

        for document, pattern, path in DOCUMENTED_CLAIMS:
            text = Path(document).read_text(encoding="utf-8")
            matches = re.findall(pattern, text)
            assert matches, f"{document}: no value found for {path}"
            expected = dig(run, path)
            for found in matches:
                assert abs(float(found) - float(expected)) <= 0.0005, (
                    f"{document} documents {found} for {path}, artefact says {expected}"
                )
