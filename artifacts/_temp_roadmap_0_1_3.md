# jnwb 0.1.3 — release roadmap (PLAN ONLY, nothing here is implemented)

**Status:** draft plan, 2026-09-09. **Branch:** `dev`, tip `07e9536`.
**Current version:** `jnwb.__version__ == '0.1.1'` (`jnwb/__init__.py:14`). `0.1.2` was never cut —
the `requires-python` fix landed on `dev` as `07e9536` and is unreleased.
**Inherited test claim:** `527 passed, 14 skipped, 0 failed` on CPython 3.14.3, per
`E:/omission/context/state/JNWB_REQUESTS.md`. **Not verified in this session** — re-running it is T0.

Everything below marked `observed` was read from the tree at `07e9536` on 2026-09-09; line numbers
are as of that read. Items sourced from the omission request file are `derived` from it and must be
re-resolved against the current tree before being fixed.

---

## 0. Release decision

Fold the unreleased `requires-python` fix into a single **0.1.3** rather than cutting 0.1.2 first.
A 0.1.2 would ship the one-line pin fix while the *policy* around it — Gate 9, the CI matrix, the
classifiers, the "3.12 only" prose — stays self-contradictory. One coherent release is cheaper to
explain and to verify.

**Cost of waiting:** PyPI `0.1.1` stays uninstallable on Python ≥3.13 and pip silently resolves to
`0.1.0` — older code — until 0.1.3 publishes. Acceptable only if 0.1.3 lands quickly. **If it slips
past ~1 week, cut 0.1.2 immediately as a metadata-only release.**

---

## 1. Python support policy — the correction that motivated this release

**Instruction (Hamm, 2026-09-09):** the earlier "3.12 only" decision was scoped to *one specific
run*, not to the library. It was over-generalized into repository-wide policy. Corrected:

- **Declared support: `>=3.12`**, no upper ceiling.
- **Tested in CI: 3.12 and 3.14** — the floor and the current head. 3.13 is supported by
  declaration and covered by interpolation, not by its own matrix leg.

| # | Change | Site |
|---|---|---|
| P1 | Keep `requires-python = ">=3.12"` — already correct via `07e9536` | `pyproject.toml:17` |
| P2 | CI matrix → `python-version: [ "3.12", "3.14" ]` | `.github/workflows/workflow.yml:30` |
| P3 | Add classifiers for 3.13 and 3.14 beside 3.12 | `pyproject.toml:22-23` |
| P4 | **Rewrite Gate 9** — it currently enforces the *wrong* policy | `scripts/harness_gate.py:378-410` |
| P5 | "Python 3.12 only" → "Python 3.12 or newer (CI: 3.12, 3.14)" | `README.md:36`, `docs/index.md:18`, `docs/install.md` |
| P6 | Keep `.readthedocs.yaml` on 3.12 — a docs build needs one interpreter, not a matrix — but Gate 9 must stop reading that pin as evidence of a global 3.12-only policy | `.readthedocs.yaml` |

### Gate 9 is the blocker, not the prose

`check_python_target_consistency` (`scripts/harness_gate.py:378`) is titled *"Assert Python 3.12 is
the sole targeted version across metadata and CI"*. As written it:

- accepts `>=3.12` (~line 388) — so P1 already passes, **but**
- **rejects** any classifier for `3.10 / 3.11 / 3.13 / 3.14` (~line 390) → blocks P3
- **requires** the CI matrix to be literally `[ "3.12" ]` (~line 404) → blocks P2

The gate must be rewritten before P2/P3 can land, or the release fails its own harness. Proposed
replacement, renamed **`check_python_floor_consistency`**:

1. `requires-python` declares a floor of 3.12 with **no upper bound** — reject any `<3.x` clause.
2. The classifier set equals the declared-supported set `{3.12, 3.13, 3.14}` — nothing below the
   floor, nothing above the newest declared.
3. The CI matrix **contains** the floor (3.12) and the newest declared version (3.14). No longer
   required to be a singleton.
4. `.readthedocs.yaml` pins a version **within** the supported range.

The invariant worth protecting is *"declared support, classifiers, and CI agree"* — not *"3.12 is
the only version"*. The adversarial probe at `tests/test_harness_adversarial_gates.py:207` updates
in the same change and **gains a new negative case: an upper pin (`<3.13`) must be rejected** —
precisely the JNWB-001 defect the current gate failed to catch.

---

## 2. Inbound defect requests (`JNWB_REQUESTS.md`)

| ID | Severity | 0.1.3 disposition |
|---|---|---|
| JNWB-001 | blocker | **done on `dev`, unreleased** — ships in 0.1.3 (§1) |
| JNWB-002 | high | in scope — new gate (§2.1) |
| JNWB-003 | high | in scope — `rng` param + hoist (§2.2) |
| JNWB-004 | medium | in scope — resolve device once (§2.3) |
| JNWB-005 | medium | in scope — expose `n_surrogates` (§2.4) |
| JNWB-006 | low | in scope — docstring wording (§2.5) |
| JNWB-007 | high | in scope — rename + deprecating alias (§2.6) |
| JNWB-R01 | — | resolved upstream; do not re-raise |
| JNWB-R02 | — | stale claim; jnwb needs no change |

### 2.1 JNWB-002 — a foreign importable package inside the library checkout

A user project cloned inside the jnwb checkout becomes importable *ahead of itself* under an
editable install, and `.gitignore:26` hides it from `git status`. Observed consequence on the
omission side: 88 of 190 importing files silently loaded the wrong copy, disagreeing on an
anatomical label. No error was raised.

**Plan:** a gate that fails when the repository root holds any directory with an `__init__.py` other
than `jnwb/`. Cheap, exact, and it converts silent wrong-code execution into a build failure. Home:
`scripts/harness_gate.py`, beside the root-allowlist gate, wired into the ordinary CI job so it
fires on every push, not only at release.

**Check before writing:** scope it to *root-level* directories with a small allowlist, mirroring
`check_root_allowlist` (`scripts/harness_gate.py:171`), so it does not fire if `tests/` or `docs/`
ever grows an `__init__.py`. (`scripts/__init__.py` exists but is not a root directory package.)

### 2.2 JNWB-003 — `cross_area_coherence` null is unseedable and re-seeded per band

Sites: `jnwb/spectral.py:302` (signature), `jnwb/spectral.py:410` (the seed). Two independent defects:

1. `np.random.default_rng(42)` is hardcoded — no `rng`/`seed` parameter, so no caller can draw an
   independent null or record a seed in a receipt.
2. The generator is constructed **inside** the band loop, so theta/alpha/beta/low_gamma/high_gamma
   all draw the *identical* shift sequence. Per-band nulls are perfectly rank-correlated, and any
   count-of-significant-bands or multiple-comparison correction over them carries an unstated
   dependency structure.

**Plan:** follow the convention already in the library at `jnwb/statistics.py:464, :722, :762` —
`rng: Optional[np.random.Generator] = None`, defaulting to `default_rng(42)`, type-checked. Hoist
construction above the band loop. Return seed provenance in the result dict. Defect (2) is a
one-line hoist and is correct independently of (1) — land it even if the API change is deferred.

**Compatibility:** the hoist **changes numerical output** for every band after the first. A
correctness fix, not a regression, but any downstream receipt quoting a per-band p-value from
`cross_area_coherence` will not reproduce. Must be called out at the top of the changelog.

### 2.3 JNWB-004 — the device can silently change estimator mid-null

The `try: <GPU> except Exception: <CPU>` sits *inside* the per-surrogate loop, so an intermittent GPU
failure (OOM being the obvious case) yields a null that is a **mixture of two estimators**. They are
not numerically equivalent: `_welch_csd_gpu` is called with `nperseg=min(len, 4096)` and the GPU
helper's default `noverlap`, while the CPU fallback passes `noverlap=None` to
`scipy.signal.coherence` — different segment counts, therefore different coherence bias, which is
the exact quantity under test. Nothing is logged; nothing appears in the result.

**Plan:** resolve the device **once**, before the loop; fall back wholesale with a warning, never per
iteration. Record the device actually used in the returned dict. If per-surrogate fallback is
deliberately kept, at minimum count the fallbacks and return the count.

**Related sweep, in scope:** `jnwb/` holds **24 `except Exception` handlers** across 10 modules
(`spectral.py` 7, `jrsa.py` 5, `analyzers.py` 3, the rest singles/doubles). Sweep for the JNWB-004
*shape* specifically — a handler inside a loop that swaps the **estimator** rather than the control
flow. A handler that only degrades an optional import is fine and stays.

### 2.4 JNWB-005 — surrogate count silently drops 50 → 10, moving the p-value floor

`n_surr = 50; if len(lfp_area1) > 50000: n_surr = 10`. With the `(count+1)/(n+1)` estimator the
smallest attainable p-value becomes a function of input length: **1/51 = 0.0196** short,
**1/11 = 0.0909** long. At 1 kHz the branch trips at 50 s of data — so a caller testing at α = 0.05
**cannot reject** on a long recording, and the return value does not say so.

**Plan:** expose `n_surrogates` with a documented default. If a length-based reduction stays as
default behaviour, return `n_surrogates_used` and the implied p-value floor, and document the floor
in the docstring.

### 2.5 JNWB-006 — the docstring misnames the surrogate

The comment says *"phase-randomized/shuffled"*; the code does `np.roll(lfp_area2, shift)`. A circular
shift preserves the full autocorrelation and amplitude spectrum and destroys only relative
alignment — a legitimate, arguably more conservative null for cross-signal coupling, but **not**
phase randomization, and the two have different null hypotheses. Fix the wording; the code is right.

### 2.6 JNWB-007 — `paths.REPO_ROOT` names the *library's* checkout

`jnwb/paths.py` exports `REPO_ROOT`, resolved from `paths.py`'s own location, so in any consuming
project it is the jnwb checkout. Measured on the omission side: **41 live files** build paths from
it, resolving to `C:/workspace/jnwb/outputs/…` — which does not exist, so those scripts read a
missing input rather than writing into the library. That is luck, not design. One of them produces
the presence table underlying every stable-unit definition, and had never run.

**Plan:** rename to `PACKAGE_ROOT` — preferred over `JNWB_ROOT` because it says *whose* root without
repeating the package name. Keep `REPO_ROOT` as a module-level alias emitting `DeprecationWarning`,
scheduled for removal in 0.2.0. Update `__all__`, docs, and the API-set-equality fixtures.

**Consider going further:** if the only legitimate consumer is jnwb's own tests, it should be private
(`_PACKAGE_ROOT`) and dropped from `__all__` entirely. A library arguably should not export its own
checkout path at all. Decide during implementation; record the outcome in the changelog.

**Compatibility:** a public-API rename — the most user-visible change in 0.1.3.

---

## 3. CUDA and parallelization — make the story uniform and observable

Observed state:

- GPU support is **~15 inline `import cupy` / `import torch` sites** across `analyzers.py`,
  `connectivity.py`, `gpu_pca.py`, `jrsa.py`, `nam.py`, `spectral.py`, `trajectory.py` — each with
  its own availability probe and its own fallback policy.
- Two different probes coexist: bare `import cupy` (`jrsa.py:875`) and `torch.cuda.is_available()`
  (`gpu_pca.py:56`, `jrsa.py:880`, `analyzers.py:717`).
- Parallelism exists in **exactly one module**: `jnwb/jrsa.py` has `n_jobs` and a joblib
  `_parallel_map` (`jrsa.py:920`). **No other module accepts `n_jobs`** — see §4 B8, the skill
  documentation currently claims otherwise.
- Optional deps already declared: `torch`, `gpu` (`cupy-cuda12x`, `jax`), `pytest-xdist` under `test`.

| # | Change |
|---|---|
| G1 | One internal `jnwb/_backend.py`: `resolve_device(device) -> str`, `xp_for(device)`. Every module routes through it. One probe, one fallback policy, one place to test. |
| G2 | **Fallback must be observable.** A CPU fallback from a requested `cuda` emits a `RuntimeWarning` naming the reason and, where a result dict exists, records `device_used`. Silent fallback is what makes JNWB-004 dangerous. |
| G3 | Per-backend seeding is explicit — a GPU path must not consume a different RNG stream than the CPU path for the same seed, or results become device-dependent. Audit `jrsa.py` and `spectral.py` for this specifically. |
| G4 | Extend `n_jobs` beyond `jrsa`, in cost order: `statistics.cluster_permutation_test`, the `spectral.py` surrogate loop, `connectivity.py` pairwise sweeps. **Default `n_jobs=1`, not `-1`** — a library that silently saturates 24 cores inside a caller's own pool is a defect. (`jrsa.py:147` already defaults to `-1`; leave it, but document it.) |
| G5 | CI: `pytest -n auto` — xdist is already a declared test dep. Confirm no test depends on execution order first. |
| G6 | A GPU smoke test that **skips** where no CUDA device exists, and **fails loudly** when `device='cuda'` is requested, CUDA *is* present, and the CPU path silently ran anyway. |

Verification capacity here: 24 cores, ~206 GB RAM, RTX A4000 with torch/jax/cupy installed. CI has
no GPU, so G6 skips there and must be run locally before release with the receipt recorded.

---

## 4. Bloat, verbosity, and drift

Observed: **12,211 lines across 26 modules** in `jnwb/`; largest are `connectivity.py` (2038),
`jrsa.py` (1552), `statistics.py` (1249), `spectral.py` (1013). `__init__.py` is 442 lines of
re-export surface.

| # | Target | Evidence | Action |
|---|---|---|---|
| B1 | **Gate numbering is corrupt.** "Gate 9" names two different gates — `check_documented_api_matches_all:204` ("Gate 9, API Set Equality") and `check_python_target_consistency:378` ("Gate 9, Python 3.12 Target"). "Gate 2", "Gate 3", and "Gate 6" are each used twice as well. | `scripts/harness_gate.py` docstrings | Renumber once, canonically. A gate report citing "Gate 9" is currently ambiguous. Do this **before** adding the §2.1 gate. |
| B2 | **Two documentation toolchains.** `mkdocs.yml` — what `.readthedocs.yaml` actually builds — *and* `docs/conf.py` (Sphinx + `sphinx_rtd_theme`, with `docs/_static/`, `docs/_build/`, and `sphinx`/`sphinx-rtd-theme`/`myst-parser` in the `docs` extra). Only one publishes. | `.readthedocs.yaml`, `mkdocs.yml`, `docs/conf.py:56-68` | Delete the loser. **Recommend keeping mkdocs**; drop the Sphinx config, `_static/`, `_build/`, and three doc deps. Check first whether any gate or script reads `docs/conf.py` for a version string — Gate 8's docstring mentions it. |
| B3 | Scattered GPU probes | §3 G1 | Consolidation *is* deletion. |
| B4 | `except Exception` sweep | 24 handlers, 10 modules | Each becomes a specific exception, a re-raise, or a documented degrade. Bare `pass` handlers go. |
| B5 | **`.gitignore` is 130+ lines, most of it stale narrative.** It carries dated commentary about omission-side migrations, a retired `.agents/skills/` tree, a "private controlling analysis goal", a Labyrinth graph migration, and an atlas builder — none of which exist in this repository. Several comment blocks describe rules whose actual patterns were deleted, leaving orphan prose. | `.gitignore` | Reduce to the patterns that apply to *this* repo, with one line of rationale each. Move history to the changelog if it is worth keeping at all. |
| B6 | **Two `AGENTS.md` that differ.** Root `AGENTS.md` (112 lines) and `artifacts/AGENTS.md` (111) are near-duplicates with divergent titles, scope notices, and content — and the root copy still describes itself as covering *"jnwb & omission"* and cites `omission/context/PROJECT_STATE.md`, a path that does not exist in this checkout. | `AGENTS.md`, `artifacts/AGENTS.md` | One file. See §5. |
| B7 | `artifacts/scratch/scratch_readiness_test.{csv,json}` | `ls artifacts/scratch` | Delete unless a gate reads them. |
| B8 | **Skill files carry stale counts and an overclaim.** `skills/jnwb/SKILL.md:7` asserts *"All 101 exports resolve"* — the live count is **111** (`len(jnwb.__all__)`, observed). The same section claims *"pytest passes 446+ tests"* against an inherited 527, and *"`sphinx-build -W` compiles docs warning-free"* while §4 B2 proposes removing Sphinx. §3 of that file states permutation testing and pairwise channel matrices *"support `n_jobs`"* — **they do not**; only `jrsa` does. That last one actively misleads an agent into passing an unsupported kwarg. | `skills/jnwb/SKILL.md` | See §5. Skill files are doctrine-adjacent — **propose, do not edit unilaterally.** |
| B9 | Prose duplication between `README.md` and `docs/index.md` | duplicated capability tables and install blocks | One capability table, one install block, cross-linked. **Careful:** `check_public_symbols_documented` (`scripts/harness_gate.py:185`) greps `docs/*.md` for every `__all__` symbol, so deleting prose can break the gate. |

**Constraint on all of B\*:** `jnwb.__all__` is the API contract. Bloat reduction removes duplication
and dead configuration, never public symbols — except the deliberate, announced `REPO_ROOT`
deprecation in §2.6.

---

## 5. Agent-facing context: skills, tools, harness (Hamm, 2026-09-09)

**Goal:** an AI agent doing analysis with this repository should be able to establish, from tracked
files alone, *what exists, what it may call, what the invariants are, and how to verify* — with the
fewest tools that give complete coverage, explained densely enough to act on without a tutorial's
padding. The failure modes below are all `observed`.

### 5.1 What is broken now

1. **Agent definitions do not travel with the repo.** `.claude/agents/` holds `claim-verifier.md`,
   `code-auditor.md`, `sweep-runner.md`, but `.gitignore:136` ignores `.claude/`
   (`git check-ignore` confirms). A fresh clone gets **zero** agent definitions. The `.gitignore`
   comment block even argues that `settings.json` should be tracked *because* an untracked copy
   "silently resurrects the policy gap" — the same argument applies to the agents, and lost.
2. **Two divergent `AGENTS.md`** (B6), so "the operational contract" has no single referent, and the
   root copy points at an `omission/` tree that is not in this checkout.
3. **Skill files assert stale numbers and one unsupported API** (B8). A skill that says
   `n_jobs` is supported where it is not is worse than a skill that says nothing.
4. **No tool inventory.** Nothing tracked answers "what scripts exist and what do they do."
   `scripts/` holds `harness_gate.py`, `release_gate.py`, `mkdocs_version_hook.py` — discoverable
   only by reading them.
5. **Skill claims are unverified.** `tests/test_skills_validation.py` exists; determine what it
   actually asserts (frontmatter shape vs. content truth) — B8's drift got through it either way.

### 5.2 Plan

| # | Change |
|---|---|
| A1 | **Track the agent definitions.** Un-ignore `.claude/agents/*.md` (and `.claude/settings.json` if not already), keeping `.claude/settings.local.json` ignored. Mirrors the reasoning already written in `.gitignore` for settings. |
| A2 | **One `AGENTS.md`.** Keep the root file as the single operational contract; delete `artifacts/AGENTS.md` or reduce it to a one-line pointer. Strip every `omission/` reference — this repository is the library. Add a gate asserting the two do not both exist with divergent content. |
| A3 | **Make skill claims mechanically checked, not asserted.** Every count a skill states (export count, test count) either comes out of the harness or is deleted. Preferred: **delete the numbers** — a count in prose is a drift generator; `jnwb.__all__` is the source of truth and the skill should say so and stop. Extend `tests/test_skills_validation.py` to fail on any bare integer claim about exports or tests. |
| A4 | **Fix the `n_jobs` overclaim in `skills/jnwb/SKILL.md` §3** — either narrow the claim to `jrsa`, or land §3 G4 first and make the claim true. Prefer landing G4; then the skill is correct and more useful. |
| A5 | **Add one tool-inventory section**, not a new file — inside `AGENTS.md`: each entry `script → what it asserts → how to run it → what a pass means`. Three scripts, three rows. |
| A6 | **Fewest tools, complete coverage.** Audit the 8 skills against the 26 modules for gaps and overlaps. The router (`skills/jnwb`) plus seven domain skills is a defensible shape; the test is whether every public module maps to exactly one skill. Report the mapping; merge or split only where a module is unreachable or double-claimed. |
| A7 | **Condensed-tutorial density.** Each domain skill should carry: trigger, the API surface it owns (symbol names only, no prose restatement of docstrings), the invariants that are easy to get wrong, and one runnable minimal example. That is roughly the current shape at ~45 lines each — **the skills are not too verbose; they are too stale.** Correctness first, length second. |

**Authority note:** skill files are doctrine-adjacent. A3/A4/A6/A7 are **proposals**. Land them only
with Hamm's explicit approval, per the standing amendment rule. B8's stale counts are factual
defects and the least controversial place to start.

### 5.3 Thinnest possible root

Tracked root files are already minimal — `.gitignore`, `.readthedocs.yaml`, `AGENTS.md`,
`CHANGELOG.md`, `CLAUDE.md`, `LICENSE`, `README.md`, `mkdocs.yml`, `pyproject.toml`. **No build
output is tracked** (verified: zero tracked files under `site/`, `_build/`, `dist/`, `docs/_build/`).

The bloat is in what the *allowlist tolerates on disk*. `ALLOWED_ROOT_DIRS`
(`scripts/harness_gate.py:160`) admits 21 directories including `_build`, `site`, `dist`,
`.lab_bundle_build`, `.pytest_cache`, `jnwb.egg-info`, `.cursor`, `.gemini`. Present on disk right
now: `_build` (112 files), `site` (195), `dist` (2), `.lab_bundle_build` (14), and `.cursor` /
`.gemini` — **both empty**. The allowlist was widened to accept build detritus rather than keeping
the root thin.

| # | Change |
|---|---|
| R1 | Remove `.cursor/` and `.gemini/` — empty, and drop them from the allowlist. Two fewer things an agent must decide are irrelevant. |
| R2 | Point build output at one ignored directory (`_build/` or `build/`), not three (`_build`, `site`, `docs/_build`). `mkdocs.yml` `site_dir` and any Sphinx output follow; §4 B2 removes one of them outright. |
| R3 | Shrink `ALLOWED_ROOT_DIRS` to what is *legitimately expected*: `jnwb tests examples docs skills scripts artifacts .git .github .claude` plus a short, commented tool-noise set. Drop the entries that only exist because someone hit the gate. |
| R4 | **Split the allowlist into two tiers** — *tracked-source* dirs (fail on anything unexpected) and *tolerated-ephemeral* dirs (must be gitignored, else fail). Today a stray tracked directory named `site` would pass. Also worth noting: the allowlist still lists `omission` as an allowed root dir, which is exactly the shadowing hazard §2.1 exists to prevent — the two must be reconciled. |
| R5 | `CLAUDE.md` (83 lines) is the agent's first read. Confirm it still describes *this* repository after §5 lands, and that it points at one `AGENTS.md`, not two. |

**Target:** a fresh clone's root shows source, docs, tests, harness, agent context — and nothing an
agent has to classify before it can start.

---

## 6. Logo

**Asset:** `C:/Users/nejath/Downloads/jnwblg.png` — 1,701,389 bytes, 2026-09-09 (observed).

Pattern to match, jaxfne (`E:/repos/jaxfne`):

- `README.md:1-3` — a centered banner **above** the badge block:
  `<img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">`.
  An absolute `raw.githubusercontent.com` URL, because relative paths do not render on PyPI.
- `mkdocs.yml:81` — `theme.logo: assets/jaxfne-img.png`, a separate smaller mark for site chrome.

| # | Change |
|---|---|
| L1 | **Downsize before committing.** 1.7 MB is ~180× the existing `docs/assets/jnwb-logo.png` (9,389 bytes). Resample to ≤400 px wide, target <60 KB. Do not commit the raw Downloads file. |
| L2 | Commit as `docs/assets/jnwb-itxt.png`, matching jaxfne's `-itxt` naming. Note `.gitignore` already whitelists `docs/assets/*.png` against the global `*.png` ignore — verify the add is not silently dropped. |
| L3 | `README.md`: insert the centered `<p align="center">` banner as **lines 1-3**, above the badge block, `width="200"`, absolute URL `https://raw.githubusercontent.com/HNXJ/jnwb/main/docs/assets/jnwb-itxt.png`. |
| L4 | `mkdocs.yml:16` (`theme.logo`) and `:17` (`favicon`): **recommend replacing the theme logo and favicon too**, so PyPI, GitHub, and the docs site show one identity. Regenerate the favicon from the same source. |
| L5 | `docs/index.md`: same banner at top, using the **relative** path `assets/jnwb-itxt.png` — mkdocs resolves relative; the absolute URL is a PyPI/GitHub-only workaround. |
| L6 | Verify: renders on GitHub *and* in the built PyPI long description; `mkdocs build --strict` passes — RTD sets `fail_on_warning: true`, so a missing asset is a hard build failure. |

---

## 7. Execution order

Dependency order, not importance. **T0 gates everything.**

| Step | Work | Gate |
|---|---|---|
| T0 | Re-run the full suite on the current tree, 3.12 **and** 3.14. Record both receipts. | Establishes the real baseline — `527 passed` is inherited, not verified here. |
| T1 | §4 B1 — renumber gates canonically | `pytest tests/test_harness_adversarial_gates.py` |
| T2 | §1 P2-P6 — rewrite Gate 9 → `check_python_floor_consistency`, then matrix, classifiers, prose | adversarial gates, incl. the new "upper pin rejected" probe |
| T3 | §2.1 — foreign-package gate + its probe; reconcile with §5.3 R4 | new gate test |
| T4 | §2.6 — `REPO_ROOT` → `PACKAGE_ROOT` + deprecating alias | API-set-equality gate, full suite |
| T5 | §2.2-2.5 — `cross_area_coherence`: hoist rng, add `rng`, resolve device once, expose `n_surrogates`, fix docstring | spectral tests + a **new** test asserting per-band nulls are not identical |
| T6 | §3 G1-G4 — `_backend.py`, observable fallback, per-backend seeding, `n_jobs` | full suite + local GPU smoke (G6) |
| T7 | §5 — agent context: A1, A2, A5 (tracked, uncontroversial), then A3/A4/A6/A7 **on Hamm's approval** | `tests/test_skills_validation.py`, extended |
| T8 | §4 B2, B5, B7, B9 + §5.3 R1-R5 — bloat and root | `mkdocs build --strict`, API-completeness gate, root-allowlist gate, full suite |
| T9 | §6 — logo | README render check + `mkdocs build --strict` |
| T10 | Release: bump `jnwb/__version__` to `0.1.3`, write the CHANGELOG entry (lead with the `REPO_ROOT` rename and the `cross_area_coherence` numerical change), tag, build, `twine check`, publish | all gates, both interpreter legs |
| T11 | Reply into `JNWB_REQUESTS.md` with per-ID disposition and landing commits; delete this file | — |

**Do not batch T4, T5, and T6 into one commit** — each changes numerical output or the public API,
and they must be independently bisectable.

---

## 8. Out of scope for 0.1.3

- Widening CI to 3.13 (see the open decision below).
- Any change to jnwb's scientific defaults not named above.
- Anything under `omission/` — this is a library release; the consumer-side repairs already landed
  there.
- Removing `REPO_ROOT` outright — deprecates in 0.1.3, removes in 0.2.0.

## 9. Unresolved — needs Hamm

1. **CI matrix:** floor+head `[3.12, 3.14]` (recommended) vs full `[3.12, 3.13, 3.14]` (§1).
2. **Classifier set:** declare 3.12/3.13/3.14, or only what CI tests (3.12/3.14)? The plan assumes
   classifiers follow *declared support*, i.e. all three (§1 P3).
3. **`PACKAGE_ROOT` vs dropping the export entirely** (§2.6).
4. **Sphinx removal** (§4 B2) — deletes a second, currently unpublished doc toolchain.
5. **Theme logo replacement** (§6 L4) — README only, or README + site + favicon?
6. **Skill edits** (§5.2 A3/A4/A6/A7) — doctrine-adjacent; proposals pending explicit approval.
7. **`omission` in `ALLOWED_ROOT_DIRS`** (§5.3 R4) — the allowlist permits exactly the directory
   §2.1's new gate would reject. One of the two must give.

---

*Temporary planning artifact (`_temp_` prefix). Superseded by the 0.1.3 CHANGELOG entry once the
release lands; delete at T11.*
