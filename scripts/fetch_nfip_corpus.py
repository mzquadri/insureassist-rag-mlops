"""
Fetch the NFIP Standard Flood Insurance Policy forms from the authoritative eCFR API.

    python -m scripts.fetch_nfip_corpus

The three SFIP forms are published as appendices to 44 CFR Part 61. They are works of the
United States Government prepared by FEMA, so under 17 U.S.C. 105 they carry no copyright
and may be redistributed. `docs/DATA.md` records that basis per document.

Why fetch rather than commit a scrape blindly: the extraction is the part that has to be
reproducible. This script pins the eCFR issue date, converts the XML to plain text by one
documented set of rules, and writes a manifest containing the SHA-256 of both the upstream
XML and every extracted file. Re-running it on the same issue date must reproduce identical
bytes, and the tests assert exactly that.

The build is offline by default. The corpus and its manifest are committed, so nothing in
the test suite, the ingestion path, or CI ever contacts eCFR. Pass --check to verify the
committed files still match their recorded hashes without re-downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

#: Pinned eCFR issue date. Bumping this is a deliberate corpus change: it alters the
#: document hashes, which invalidates the recorded chunk IDs and therefore the labels.
ECFR_ISSUE_DATE = "2026-06-22"
ECFR_API = (
    f"https://www.ecfr.gov/api/versioner/v1/full/{ECFR_ISSUE_DATE}/title-44.xml?part=61"
)

from src.paths import CORPUS_DIR

MANIFEST_PATH = CORPUS_DIR / "manifest.json"

#: Appendix name in the XML -> (document id, human title, the form's short name).
FORMS = {
    "Appendix A(1) to Part 61": (
        "nfip-sfip-dwelling",
        "Standard Flood Insurance Policy - Dwelling Form",
        "Dwelling Form",
    ),
    "Appendix A(2) to Part 61": (
        "nfip-sfip-general-property",
        "Standard Flood Insurance Policy - General Property Form",
        "General Property Form",
    ),
    "Appendix A(3) to Part 61": (
        "nfip-sfip-rcbap",
        ("Standard Flood Insurance Policy - Residential Condominium Building "
         "Association Policy"),
        "RCBAP",
    ),
}

CANONICAL_URL = "https://www.ecfr.gov/current/title-44/chapter-I/subchapter-B/part-61"
LEGAL_BASIS = (
    "17 U.S.C. 105 - work of the United States Government, not subject to copyright "
    "protection in the United States. Authored by FEMA and promulgated as 44 CFR Part 61."
)

#: Tags that start a new block of text. Everything else is inline.
_BLOCK_TAGS = {"HEAD", "HD1", "HD2", "HD3", "HD4", "P", "FP", "CITA"}
#: Headings, recorded so the extracted text keeps the form's own section structure.
_HEADING_TAGS = {"HEAD", "HD1", "HD2", "HD3", "HD4"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _normalise_inline(text: str) -> str:
    """Collapse the XML's incidental line wrapping into single spaces.

    The eCFR XML wraps paragraphs across source lines for readability. That wrapping is a
    formatting artefact of the transport format, not part of the regulation, so it is
    normalised away. Wording, punctuation, capitalisation and every numeric value are left
    exactly as published - this must never paraphrase.
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_blocks(appendix: ET.Element) -> list[str]:
    """Turn one appendix element into an ordered list of text blocks."""
    blocks: list[str] = []
    for element in appendix.iter():
        if element.tag not in _BLOCK_TAGS:
            continue
        text = _normalise_inline("".join(element.itertext()))
        if not text:
            continue
        if element.tag in _HEADING_TAGS:
            # Marked so chunking and any later structural work can see section boundaries.
            blocks.append(f"## {text}")
        else:
            blocks.append(text)
    return blocks


def render_document(blocks: list[str]) -> str:
    """One block per paragraph, blank line between blocks, trailing newline.

    A stable, boring representation: offsets recorded against this text stay valid as long
    as the extraction rules do not change.
    """
    return "\n\n".join(blocks) + "\n"


def fetch_xml() -> bytes:
    request = urllib.request.Request(
        ECFR_API,
        headers={"User-Agent": "insureassist-rag-mlops corpus build (+github.com/mzquadri)"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # fixed https URL, not user input
        return response.read()


def build(check_only: bool = False) -> int:
    if check_only:
        return verify()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_xml()
    root = ET.fromstring(raw)

    appendices = {
        element.get("N"): element
        for element in root.iter()
        if element.get("TYPE") == "APPENDIX"
    }

    documents = []
    for appendix_name, (doc_id, title, form_name) in FORMS.items():
        element = appendices.get(appendix_name)
        if element is None:
            raise SystemExit(f"{appendix_name!r} not found in the eCFR response")

        text = render_document(extract_blocks(element))
        path = CORPUS_DIR / f"{doc_id}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")

        documents.append({
            "document_id": doc_id,
            "title": title,
            "form": form_name,
            "cfr_citation": f"44 CFR Part 61, {appendix_name}",
            "filename": path.name,
            "source_url": CANONICAL_URL,
            "api_url": ECFR_API,
            "ecfr_issue_date": ECFR_ISSUE_DATE,
            "legal_basis": LEGAL_BASIS,
            "redistribution_permitted": True,
            "attribution": (
                "Federal Emergency Management Agency, 44 CFR Part 61, "
                f"{appendix_name} (eCFR issue {ECFR_ISSUE_DATE})."
            ),
            "sha256": sha256_file(path),
            "characters": len(text),
            "words": len(text.split()),
        })

    manifest = {
        "corpus_id": "nfip-sfip",
        "description": (
            "NFIP Standard Flood Insurance Policy forms as published in 44 CFR Part 61."
        ),
        "retrieved_at": datetime.now(timezone.utc).date().isoformat(),
        "ecfr_issue_date": ECFR_ISSUE_DATE,
        "source_xml_sha256": sha256_bytes(raw),
        "extraction": {
            "method": "scripts/fetch_nfip_corpus.py",
            "input_format": "eCFR versioner API XML (title-44, part 61)",
            "output_format": "UTF-8 plain text, LF line endings",
            "rules": [
                "Block elements HEAD/HD1-HD4/P/FP/CITA become paragraphs in document order.",
                "Headings are prefixed with '## ' to preserve the form's own structure.",
                ("Runs of whitespace inside a block collapse to a single space; the XML "
                 "wraps lines for transport and that wrapping is not part of the regulation."),
                "Blocks are joined with a blank line; the file ends with a newline.",
                "No text is paraphrased, reordered, summarised, or otherwise altered.",
            ],
        },
        "documents": documents,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    for doc in documents:
        print(f"  {doc['document_id']:32} {doc['words']:>6} words  {doc['sha256'][:16]}")
    print(f"Wrote {len(documents)} documents and {MANIFEST_PATH}.")
    return 0


def verify() -> int:
    """Re-hash the committed corpus against the manifest. No network access."""
    if not MANIFEST_PATH.exists():
        print("No manifest; run without --check to build the corpus.", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = 0
    for doc in manifest["documents"]:
        path = CORPUS_DIR / doc["filename"]
        if not path.exists():
            print(f"  MISSING {path}")
            failures += 1
            continue
        actual = sha256_file(path)
        ok = actual == doc["sha256"]
        print(f"  {'ok  ' if ok else 'FAIL'} {doc['document_id']:32} {actual[:16]}")
        failures += not ok

    print("Corpus matches the manifest." if not failures else f"{failures} mismatch(es).")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify committed files against the manifest without downloading",
    )
    raise SystemExit(build(check_only=parser.parse_args().check))
