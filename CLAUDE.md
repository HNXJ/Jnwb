# CLAUDE.md — jnwb

Generic, dataset-agnostic NWB (Neurodata Without Borders) analysis library. Project-specific
doctrine (paradigms, corpus specifics, manuscript rules) lives under each project folder's own
`CLAUDE.md` — e.g. [`omission/CLAUDE.md`](omission/CLAUDE.md) for the `omission` example
project — and adds to, never overrides, what's here.

## Phase (2026-08-24): repository normalization closed, analysis phase active

Repository normalization ended 2026-08-24 by Hamm's explicit instruction. The repo-wide
no-commit freeze (originally declared 2026-08-19 → 2026-09-28) is **superseded**, not merely
suspended: its own commit history (`310a8ac`) carries no rationale beyond a scheduling pause, and
Hamm's 2026-08-24 instruction to begin scientific analysis is a newer, explicit instruction that
supersedes a project-level scheduling decision per this project's own precedence rule. Routine
commit/push activity may resume for analysis-phase work; nothing about this reinstates unrelated
normalization/cleanup work (see "Analysis-only" doctrine in `omission/CLAUDE.md`).

**The `jnwb/` edit freeze is LIFTED (2026-09-09, Hamm's explicit instruction).** It is
superseded, not suspended, and must not be reinstated by inference.

The freeze rested on a premise: that the package was stable, so the cost of editing it exceeded
the benefit. That premise was an **AI agent's assertion, never a verified claim** — and it is
false while defects remain open. `JNWB_REQUESTS.md` carried seven, one of them a blocker that
made 0.1.1 uninstallable on every current interpreter. A freeze justified by stability cannot
survive evidence that the package is not stable; a stale freeze protects defects rather than
consumers. Hamm's ruling: *"when there is even one open issue, freeze is stale."*

`jnwb/` is therefore editable under ordinary discipline — smallest justified change, invariants
preserved, receipts for every claim. What replaces the freeze is not permission to churn: it is
the requirement that each change to `jnwb/` names the defect it closes.

Two protections that the word "freeze" was doing double duty for **remain in full force**, and
neither depended on the edit freeze:

1. **The layering invariant** (tripwire 3): `jnwb/` imports nothing from any project folder.
   This is what `tests/test_jnwb_frozen_boundary.py` actually asserts — zero `omission/` imports,
   that `import jnwb` succeeds with `omission/` blocked from `sys.path` entirely, and that every
   `jnwb.__all__` name resolves. Note the file's own docstring frames itself as enforcing the
   freeze; it never did. It enforces layering, which is a different and still-live invariant.
2. **API stability as a contract, not a prohibition**: `jnwb.__all__` is what consumers depend
   on. Breaking changes are allowed, but must be deliberate, announced in `CHANGELOG.md`, and
   carry a deprecation path where one is possible (see `paths.PACKAGE_ROOT`, 0.1.3).

All omission-related work — scripts, figures, evidence, tests, docs — still stays inside
`omission/`. That is a layering rule, not a consequence of the freeze.

**Still protected, unrelated to phase**: paths that were dirty/uncommitted as of 2026-08-22 —
pre-existing concurrent figure/script work under `omission/context/figures/` and
`omission/scripts/`, plus `omission-data/SKILL.md` — remain untouched by any Claude session:
do not move, stage, revert, stash, or commit them. This protection is not tied to the
normalization effort; it protects concurrent human work regardless of what phase this repo is in.

## Where truth lives

| Question | Source | Never |
|---|---|---|
| What is in the public API | `jnwb/__init__.py`'s `__all__` | a symbol list remembered from any document, including this one |
| What was actually computed | the receipt named beside the number | a summary of it |

## Tripwires

1. **No empirical value in any output that no script computed from data.** Hardcoded values
   are permitted only for visual/task constants or output explicitly marked synthetic.
2. **Take the logarithm last.** Average power, divide by baseline, `10·log10` once. Never
   average decibels — it biases each site by its own noisiness.
3. **`jnwb/` does not import from any project folder** (e.g. `omission/`). The dependency runs
   one way: projects depend on `jnwb`, never the reverse. **There are now zero exceptions** —
   do not add one without discussing the layering first. (`jrsa.py`'s former exception, a lazy
   import of `phase_slope_index`, was removed 2026-08-23 when `connectivity.py` promoted to
   `jnwb/connectivity.py`. `addressing.py`'s — a call-time import of
   `omission.jnwb_ext.sequence_layout.parse_probe_areas` — was removed 2026-09-03: an optional
   import made jnwb resolve probe areas differently depending on whether omission happened to
   be importable, so merely installing a project package changed which cortical area a unit was
   assigned to. `addressing.py` now carries no area vocabulary at all: it splits a probe label
   on comma or slash, trims whitespace, and returns every label exactly as the file wrote it.
   In particular `DP` is *not* aliased to `V4` — that alias collapsed a `"DP/V4"` probe to
   `('V4','V4')` — and casing is *not* normalized; whether two spellings or two names denote
   one area is a corpus convention, not something generic addressing decides. A project that
   knows its own convention normalizes on its own side.)

   The invariant this protects: **`jnwb` gives identical scientific behaviour whether a project
   package is installed or absent.** Installing one must never be the mechanism that keeps
   `jnwb` correct.
4. **Preserve module- and array-level invariants**: units, coordinate frames, timestamps,
   sample rates, 0- vs 1-indexing do not change silently across a `jnwb` function boundary.

## Working agreements

- Preserve originals; write revisions as new files.
- Do not commit or push unless asked.

## Skills

Load the skill before doing the work; do not reinvent its contents. See a project folder's own
`CLAUDE.md` for its task-scoped skills (e.g. `omission/CLAUDE.md`).
`numerical-computing` · `biophysical-modeling`
