# Corpus and data provenance

This repository contains two kinds of document, kept deliberately separate.

| Location | What | Rights |
|---|---|---|
| `data/corpus/` | Real NFIP flood insurance policy forms, from the CFR | US Government work, no copyright (17 U.S.C. 105) |
| `data/*.md`, `data/qa_testset.jsonl` | Synthetic sample policies and questions | Written for this project; MIT with the code |

**The MIT licence covers the code only.** It does not extend to the government documents in
`data/corpus/`. See [`NOTICE`](../NOTICE).

## The NFIP corpus

The National Flood Insurance Program's Standard Flood Insurance Policy is unusual and
useful: the actual policy wording is not a private contract but is **enacted as federal
regulation**, published as appendices to 44 CFR Part 61. That makes it real insurance
policy text that can be redistributed without permission.

| Document ID | CFR citation | Form | Words | Characters | SHA-256 (first 16) |
|---|---|---|---|---|---|
| `nfip-sfip-dwelling` | 44 CFR Pt. 61, App. A(1) | Dwelling Form | 12,603 | 74,960 | `c88e41a8bcdca3ac` |
| `nfip-sfip-general-property` | 44 CFR Pt. 61, App. A(2) | General Property Form | 11,262 | 67,255 | `7b33a80ef173426d` |
| `nfip-sfip-rcbap` | 44 CFR Pt. 61, App. A(3) | RCBAP | 11,774 | 70,393 | `229d7bcc62bd0dce` |
| **Total** | | **3 documents** | **35,639** | **212,608** | |

Full per-document metadata — source URL, API URL, eCFR issue date, retrieval date, legal
basis, attribution string, hash, counts — is in
[`data/corpus/manifest.json`](../data/corpus/manifest.json).

- **Canonical source:** https://www.ecfr.gov/current/title-44/chapter-I/subchapter-B/part-61
- **Pinned eCFR issue date:** `2026-06-22`
- **Legal basis:** 17 U.S.C. 105 — a work of the United States Government is not subject to
  copyright protection. Redistribution, modification and commercial use are all permitted.
- **Attribution:** not legally required; recorded per document anyway.

### Why these three forms

They are near-duplicates *by design*. All three share a skeleton — Agreement, Definitions,
Property Covered, Property Not Covered, Exclusions, Deductibles, General Conditions — but
differ in what they insure and in several specific provisions. The RCBAP, for example,
carries a **coinsurance** article that the Dwelling Form does not have.

That makes them an unusually good retrieval benchmark. A retriever cannot succeed by topic
matching alone: for many questions, three passages across three documents look almost
identical and only one is correct. Those are the hard negatives, and they occur naturally
rather than being manufactured.

## How the text was produced

`scripts/fetch_nfip_corpus.py` fetches the eCFR versioner API for Title 44 Part 61 at a
pinned issue date and converts the XML to plain text.

```bash
python -m scripts.fetch_nfip_corpus            # rebuild from eCFR
python -m scripts.fetch_nfip_corpus --check    # verify committed files, offline
```

**Transformation applied.** The source is XML, so producing plain text necessarily changes
formatting. Exactly these rules apply:

1. Block elements (`HEAD`, `HD1`–`HD4`, `P`, `FP`, `CITA`) become paragraphs in document
   order.
2. Headings are prefixed with `## ` so the form's own section structure survives.
3. Whitespace runs *inside* a block collapse to a single space. The eCFR XML wraps lines
   for transport; that wrapping is not part of the regulation.
4. Blocks are joined with a blank line. Files end with a newline and use LF endings.
5. **Nothing is paraphrased, reordered, summarised, or reworded.** Wording, punctuation,
   capitalisation and every numeric value are exactly as published.

The result is deterministic: the same issue date produces byte-identical files, which is
asserted by the test suite against the recorded hashes.

### Changing the pinned date is a breaking change

Bumping `ECFR_ISSUE_DATE` re-downloads the regulation as amended. That changes document
hashes, which changes chunk offsets, which changes content-derived chunk IDs — and every
ground-truth label points at a chunk ID. A corpus bump therefore invalidates the labels and
the reference run, and all three must be regenerated together.

## Offline by default

The corpus and manifest are committed. Nothing in the ingestion path, the test suite, or CI
contacts eCFR. `scripts/fetch_nfip_corpus.py` is the only code that touches the network,
and it is run by hand.

## What this corpus supports

At 600-character chunks with 100 overlap it yields **426 chunks** across 3 documents
(Dwelling 150, General Property 135, RCBAP 141). Retrieval at `TOP_K=5` therefore returns
**1.17%** of the index per query, against **57%** for the synthetic sample corpus — which is
the whole reason for its existence. Retrieval metrics measured here mean something; the same
metrics on the synthetic corpus did not.

Chunk identity is stable: `chunk_id = "{document_id}#{sha256(document_id|start|text)[:12]}"`,
and the Qdrant point ID is `uuid5` of that. All 426 IDs are unique, identical across runs,
and every chunk's offsets reproduce its text exactly from the source file.

It remains a single jurisdiction and a single peril. Results describe how the retriever
performs on US flood policy wording, not on insurance documents generally.
