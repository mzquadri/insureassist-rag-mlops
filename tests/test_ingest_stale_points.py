"""A collection must mirror the corpus and the chunking, not accumulate both.

Upserting is idempotent for an unchanged corpus: a point ID is a uuid5 of a
content-derived chunk ID, so a second run rewrites the same points. That was already
tested, and it is not sufficient. Whatever a previous run wrote and the current one does
not write stays, and the collection was recreated only when the embedding dimension
changed, which a chunking change does not touch.

Measured against a real Qdrant before this was written: ingesting the three NFIP documents
at 600/100 after a run at 800/120 left the 314 old points in place and added 426 new ones,
and the collection then answered queries from 740 points drawn from two different
chunkings. The closing line printed the total and asserted nothing about it.

These tests use an in-memory stand-in. The scroll and delete contract is small, and the
real path is covered by the CI job that ingests into a live Qdrant.
"""

import types

import pytest

from src.ingest import remove_stale_points


class FakeCollection:
    """The slice of the Qdrant client that stale-point removal touches."""

    def __init__(self, ids):
        self.points = list(ids)
        self.deleted: list = []
        self.scroll_calls = 0

    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls += 1
        start = offset or 0
        page = self.points[start:start + limit]
        nxt = start + limit if start + limit < len(self.points) else None
        return [types.SimpleNamespace(id=i) for i in page], nxt

    def delete(self, *, collection_name, points_selector):
        removed = list(points_selector.points)
        self.deleted.extend(removed)
        self.points = [p for p in self.points if p not in removed]


def test_nothing_is_removed_when_the_collection_already_matches():
    store = FakeCollection(["a", "b", "c"])
    assert remove_stale_points(store, "c", {"a", "b", "c"}) == 0
    assert store.deleted == []
    assert sorted(store.points) == ["a", "b", "c"]


def test_points_from_an_earlier_chunking_are_removed():
    """The 800/120 -> 600/100 case, in miniature."""
    old, new = {"old1", "old2", "old3"}, {"new1", "new2"}
    store = FakeCollection(old | new)          # after the upsert, both generations present
    assert remove_stale_points(store, "c", new) == 3
    assert sorted(store.points) == ["new1", "new2"]
    assert sorted(store.deleted) == ["old1", "old2", "old3"]


def test_a_shrinking_corpus_drops_the_removed_document():
    store = FakeCollection(["doc-a-1", "doc-a-2", "doc-b-1"])
    assert remove_stale_points(store, "c", {"doc-a-1", "doc-a-2"}) == 1
    assert store.points == ["doc-a-1", "doc-a-2"]


def test_an_empty_collection_is_not_an_error():
    store = FakeCollection([])
    assert remove_stale_points(store, "c", {"a"}) == 0


def test_every_point_removed_when_the_whole_corpus_is_replaced():
    store = FakeCollection(["x", "y"])
    assert remove_stale_points(store, "c", {"p", "q"}) == 2
    assert store.points == []


def test_removal_pages_through_a_collection_larger_than_one_scroll():
    """SCROLL_BATCH is 1000; a collection must not be judged on its first page alone."""
    from src.ingest import SCROLL_BATCH

    keep = {f"keep-{i}" for i in range(10)}
    stale = [f"stale-{i}" for i in range(SCROLL_BATCH + 250)]
    store = FakeCollection(sorted(keep) + stale)
    assert remove_stale_points(store, "c", keep) == len(stale)
    assert store.scroll_calls > 1, "should have needed more than one page"
    assert sorted(store.points) == sorted(keep)


def test_deletions_are_batched_rather_than_sent_one_by_one():
    from src.ingest import BATCH_SIZE

    calls = []

    class Counting(FakeCollection):
        def delete(self, *, collection_name, points_selector):
            calls.append(len(points_selector.points))
            super().delete(collection_name=collection_name, points_selector=points_selector)

    store = Counting([f"s{i}" for i in range(BATCH_SIZE * 2 + 5)])
    remove_stale_points(store, "c", set())
    assert max(calls) <= BATCH_SIZE, f"a delete carried {max(calls)} ids"
    assert sum(calls) == BATCH_SIZE * 2 + 5


@pytest.mark.parametrize("current", [set(), {"a"}, {"a", "b"}, {"a", "b", "zzz"}])
def test_afterwards_the_collection_is_exactly_what_both_sides_agree_on(current):
    """The property the whole thing exists for.

    Removal deletes; it does not insert. So the collection ends as the intersection of what
    it held and what this run wrote, and an id in `current` that was never upserted, "zzz"
    below, does not appear by magic.
    """
    held = {"a", "b", "stale1", "stale2"}
    store = FakeCollection(sorted(held))
    removed = remove_stale_points(store, "c", current)
    assert set(store.points) == held & current
    assert removed == len(held - current)
