# Review

A read of this repository against its own claims, done on 2026-10-23 on Windows 11 with
CPython 3.11.14 in the repository's existing virtual environment. Every number below came
from a run recorded here.

## What it is

Retrieval-augmented question answering over the three NFIP Standard Flood Insurance Policy
forms, with citations that resolve to character offsets in the source regulation. The
distinguishing property is not the pipeline, which is conventional: it is that every
published number is re-derived from one committed artefact by a guard CI runs, and the
guard covers the numbers that look bad as well as the ones that look good.

## What was run, and what passed

Everything CI runs in its `quality` job, locally, plus the two checks CI does not have a
local equivalent for.

| Step | Command | Result |
|---|---|---|
| Lint | `ruff check src eval tests scripts` | **All checks passed** |
| Compile | `python -m compileall -q src eval tests scripts` | Clean |
| Corpus checksums | `python -m scripts.fetch_nfip_corpus --check` | **Corpus matches the manifest**, 3 of 3 documents |
| Ground truth | `python -m eval.validate_ground_truth` | **All labels resolve.** 50 questions, 32 answerable, 18 unanswerable, split dev 23 / test 27 |
| Docs vs the run | `python -m eval.verify_artifacts` | **Documentation matches the reference run**, 27 claims checked |
| Tests | `pytest` | **447 passed, 9 skipped**, 456 collected, 4.0 s |
| Secret scan | `gitleaks git .` | **No leaks found**, over the whole history |

The 9 skips are all of `tests/test_docker_integration.py`, gated behind
`INSUREASSIST_INTEGRATION=1`. CI's `container-integration` job sets it, so they run
there against a real container. That is the right way round: they are skipped because
Docker is not assumed locally, not because they are broken.

**The ground-truth and artefact guards are the two best things here.** A repository that
can fail its own build for a stale sentence in its README is rare, and this one does:
`verify_artifacts` reads the prose and reconciles 27 figures against
`eval/reference_run.json`.

### One note on scope, so a future reader does not repeat the mistake

Running `ruff check .` over the whole tree reports 6 errors, and `ruff format --check .`
reports 55 files it would reformat. **Neither is a regression.** CI lints
`src eval tests scripts` only, and never runs `ruff format` at all. The 6 errors are in
notebooks under `archive/`, which are deliberately outside that scope.

Reformatting 55 files to satisfy a command the project does not run would be a large
mechanical diff with no reviewer able to see the real change inside it, so it was not
done. If formatting is ever to be enforced, it belongs in its own commit with the CI
step added in the same change, not as a side effect of a review.

## What was wrong

### 1. The website's question counts, and its test count

This is in `ZQ`, not here, and is fixed there. It is recorded here because the numbers
are this repository's.

`ZQ/src/content/insureassist-benchmark.ts` pins commit
`526495a7e4c92bedb2492af68278232ffac75c32` and then states:

| Field | Website said | This repository, at that pin **and** at HEAD |
|---|---|---|
| `questions.total` | 40 | **50** |
| `questions.answerable` | 32 | 32 (correct) |
| `questions.unanswerable` | 8 | **18** |
| `questions.dev` | 18 | **23** |
| `questions.test` | 22 | **27** |
| `tests` | 224 | **456 collected**, 447 passing offline |

The pinned commit is dated 2026-09-24 and already carried 50 labels, so the site was not
quoting an older release: it cited one commit and published another's numbers. Every
retrieval figure it quotes is correct and matches exactly, which is what made it hard to
notice. So does the corpus block: 3 documents, 314 chunks, 35,639 words.

Neither 40 nor 224 corresponds to any state of this repository that is still reachable:
the pin has 256 test functions and HEAD has 437.

### 2. CI had no `permissions` block

Four jobs, no declared token scope. In effect it was already narrow, because that is the
repository default, but in effect is not the same as declared and a change to the
repository or organisation default would widen it silently. `permissions: contents: read`
is now set at workflow level in both workflows.

For `deploy.yml` this needed a judgement rather than a copy: Azure is reached with a
service principal from `secrets.AZURE_CREDENTIALS` through `azure/login`, **not** with
OIDC federation, so `id-token: write` is not required and is deliberately absent. The
comment there says so, because moving that workflow to OIDC later without adding it would
fail with no token to exchange.

### 3. No job had a timeout

The default is six hours. A hung Qdrant service container or a stalled download would
hold a runner for an afternoon and the branch would never get a result. Observed runs sit
around four minutes, so the caps are 15, 20, 20 and 10 minutes.

This matters more than it looks. On another repository in this pass, a job with a cap set
too tight was **cancelled** mid-suite, and a cancelled run reports neither pass nor fail.
A cap should be generous enough to never fire on a healthy run and tight enough to fire
before a hung one wastes the day.

### 4. Two actions were on deprecated Node

`actions/checkout@v4` and `actions/setup-python@v5` target Node 20, which GitHub has
deprecated and is force-running on Node 24. Both moved to `@v7`, the versions
`mcp-policy-gateway` already runs green on across three Python versions. Third-party
actions were left alone: `azure/login@v3` and `azure/container-apps-deploy-action@v2` are
not mine to bump without a way to verify the result, and the deploy workflow is manual.

### 5. `container-integration` was the one job without a pip cache

It installs both requirements files, exactly like the other two, so it now gets
`cache: pip` as well.

### 6. The workflow header undercounted its own jobs

`ci.yml` opened with "CI: three jobs" while defining four. The README has always said
four. `manifests` was the one being left out.

## What was changed here

| Change | Why |
|---|---|
| `permissions: contents: read` in both workflows | Declared least privilege |
| `timeout-minutes` on all four CI jobs | A hung job should fail, not idle |
| `actions/checkout@v7`, `actions/setup-python@v7` | Clears the Node 20 deprecation |
| `cache: pip` on `container-integration` | Consistency with the other jobs |
| "three jobs" to "four jobs" | It was wrong |
| Two Mermaid diagrams | See below |
| This file | The brief asks for it |

### The diagrams

Two polished SVGs already exist, both theme-aware through `<picture>`: the architecture,
and the MLOps pipeline. Both are generated by `scripts/figures/generate_diagrams.py` from
the reference run, so they cannot drift from the system. They stay exactly as they are.

What the README lacked was a diagram that renders as text on GitHub, so two Mermaid
diagrams were added, and deliberately **not** redrawings of the SVGs, because two pictures
of one thing is how a diagram starts disagreeing with its source:

1. **The retrieval flow**, index time above query time, with the two retrievers drawn side
   by side. The split is the whole point of the design and a single arrow hides it: dense
   cosine and BM25 each contribute 20 candidates, fusion reads rank positions only, and
   the top 5 reach the generator. Those numbers are prose-only in the README otherwise.
2. **The evidence loop**, which is this repository's actual distinguishing feature and was
   drawn nowhere: corpus and labels into a fresh run, fresh run against the committed one
   through `compare_runs`, committed one against the prose through `verify_artifacts`, and
   either disagreement failing the build.

Every label was taken from `eval/reference_run.json` or from the workflow itself, not from
the prose. Strokes use the palette in `scripts/figures/portfolio_style.py` (`#2563EB`,
`#059669`, `#D97706`, `#7C3AED`, `#DC2626`, `#9CA3AF`); fills are left to Mermaid's theme
so both stay legible in GitHub's light and dark modes, which a fixed light fill would
break. Link indices were counted and checked, because an out-of-range `linkStyle` makes
Mermaid refuse to render the whole diagram and show a parse error instead.

`verify_artifacts` was re-run after the README changed and still reconciles its 27 claims.

## What was not changed, and why

- **The two stale local branches.** `fix/stale-benchmark-counts` and
  `mlflow-chunking-sweep` both contain **zero** commits that are not already on `main`:
  `main..fix/stale-benchmark-counts` is empty and `main` is 12 commits ahead of it.
  `mlflow-chunking-sweep` is at `main`'s own commit with its remote already deleted. They
  are safe to delete and were, locally.
- **The stale local `.env`.** An untracked `.env` dated 2026-08-08 sits beside an
  `.env.example` rewritten on 2026-10-21. It names `QDRANT_COLLECTION=insurance_docs`
  rather than `nfip_sfip` and an old adapter path, and it contains no credential: the one
  key-shaped line, `LANGSMITH_API_KEY`, is empty. Earlier notes recorded that this file
  broke the local test run. **It no longer does**: the suite passes with it in place, so
  that note is out of date. It is the owner's local file and was left alone.
- **The unanswerable-rejection rate of 0.000.** The README states it plainly as a
  limitation, with the threshold experiment that failed to fix it. That is the correct
  treatment of a bad result and there is nothing to fix in the documentation.
- **The GKE guide.** Still placeholder values, and the README says so in those words.
  Honest, so left.

## Guidance applied

Named because it was used.

- **[Security hardening for GitHub Actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions)**:
  the explicit `permissions` blocks, the job timeouts, and the reasoning recorded above
  for why `id-token` is absent and why third-party actions were not bumped.
- **[Google Engineering Practices](https://google.github.io/eng-practices/)**: the change
  is scoped to verified claims and CI hygiene. The 55-file reformat was explicitly
  declined as a separate concern rather than folded in.
- **[Conventional Commits](https://www.conventionalcommits.org/)**: `ci:` for the workflow
  changes, since no application behaviour moves.
- **OWASP**, via a `gitleaks` scan of the full history, which is clean.
- **PEP 8 via ruff**, at the scope the project enforces, which passes.
- **[Diátaxis](https://diataxis.fr/)**: the README is explanation and reference and
  `docs/` holds the how-to and the methodology. That split is already right and was not
  restructured.
