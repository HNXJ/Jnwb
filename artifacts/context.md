# `jnwb` — Context for Automated Agents

> **Read this first.** This file is the single orientation point for an automated agent working
> in this repository: what `jnwb` is, where truth lives, which entry points are authoritative,
> and which commands verify a claim. It is deliberately self-contained — an agent that reads
> only this file should be able to act correctly, and should know where to look for the rest.
>
> Nothing here is tool-specific. It assumes only an agent that can read files, run commands, and
> carry notes between turns — a terminal coding assistant, a sandboxed research harness, or any
> system with a persistent memory or skill layer.

---

## 1. What this package is

`jnwb` is a **generic, dataset-agnostic** analysis library for Neurodata Without Borders (NWB)
electrophysiology: spiking, local field potentials, time-frequency representations, directed
connectivity, decoding, statistics, and figures. Its public surface is plain arrays and plain
keyword arguments.

The boundary that matters most: **`jnwb` never imports from a project or dataset package.** The
dependency runs one way — projects depend on `jnwb`, never the reverse. The invariant this
protects is that **`jnwb` behaves identically whether or not any given project package is
installed.** Installing something must never be the mechanism that keeps this library correct.

Consequences an agent must respect:

- No experiment-specific condition tokens, session labels, area vocabularies, or manuscript
  findings inside `jnwb/`, `docs/`, `skills/`, or `tests/`.
- Domain conventions that belong to one corpus (how two spellings of an area name relate, which
  label is a mislabel) are the *project's* business, not this library's. When a generic function
  is asked to encode one, that is a signal to stop, not to add a special case.
- A gate enforces this mechanically; see §5.

## 2. Where truth lives

Authority resolves in strict descending order. When two sources disagree, the higher one wins,
and an unresolved material conflict is a reason to stop and surface rather than to pick a side.

| Rank | Source | Example |
|---|---|---|
| 1 | **Direct empirical receipt** | the CSV/JSON/artifact written by verified code, named beside the claim |
| 2 | **Live repository state** | what `git status`, the file on disk, and a fresh import actually say |
| 3 | **Structured project state** | manifests, index tables, machine-readable logs |
| 4 | **Narrative prose** | this file, docstrings, doctrine documents, conversation history |

Two corollaries that catch most real errors:

- **`execution != verification`.** A command exiting `0` verifies nothing about content. Read the
  code that produced a number and confirm it traces to a computation on real data — not a
  literal, not a random draw, not a fallback branch.
- **A registry can be stale without erroring.** Any file that points at other files — a manifest,
  a skill index, a symbol list, *this table* — can name something that no longer exists. Resolve
  entries against disk before trusting them. Never quote a path, count, flag, or API name from
  memory; re-derive it.

## 3. Authoritative entry points

Resolve each of these against disk before relying on it.

| Question | Read |
|---|---|
| What is in the public API | `jnwb/__init__.py`'s `__all__`, and `docs/api.md` for signatures |
| How do I do X with this library | `skills/` — the canonical task-routed procedures |
| What are the operational rules | `AGENTS.md` (repository contract), `artifacts/AGENTS.md` (generalized policy kernel) |
| How do I extend it without breaking it | `docs/11_extending_and_development.md` |
| Worked recipes and accumulated gotchas | `docs/memory.md` |
| What a topic actually means | the twelve numbered guides in `docs/` |

## 4. Invariants that outrank convenience

1. **No empirical value that no code computed from data.** Hardcoded numbers are permitted only
   for visual or task constants, or for output explicitly marked synthetic. Missing data fails
   loudly or is labelled synthetic in the output — never silently filled.
2. **Take the logarithm last.** Form the per-unit ratio, aggregate the *ratios*, then convert to
   decibels exactly once. Averaging decibels is a Jensen error that biases every unit by its own
   noisiness. `aggregate_to_db` exists to make the correct order the only reachable one.
3. **One-way dependency.** See §1. There are no exceptions.
4. **Units, frames, and indexing do not change silently** across a function boundary — sample
   rates, time bases, coordinate frames, 0- vs 1-indexing. An intentional break is stated at the
   change site, not discovered downstream.
5. **A rule that is easier to retype than to reuse will be retyped**, and the copy will drift
   from its docstring without anyone noticing. Prefer calling the library function; if its shape
   prevents that, widening the shape is the fix, not forking the rule.

## 5. Verifying a claim

Run these before asserting the repository is healthy. Report what the command printed, not what
it was expected to print.

```bash
python scripts/harness_gate.py                 # operational and boundary gates
mkdocs build --strict                          # documentation builds warning-free
python -m sphinx -W -b html docs docs/_build/html
python -m pytest -n auto tests/                # full suite
```

The gates check, among other things, that the one-way dependency holds, that no dataset-specific
token has leaked into generic code, and that **every** name in `jnwb.__all__` is documented. A
change that breaks the boundary fails a gate before it reaches human review.

**A caution about counts.** Test totals and symbol counts quoted in prose go stale, and a suite
whose result depends on which optional packages happen to be importable is not a stable baseline.
Re-run and read the output; treat any number written in a document as a hypothesis.

## 6. Working agreements

- Make the smallest change that reaches a passing state; preserve unrelated invariants; no
  drive-by edits.
- Preserve originals; write revisions as new files where a rewrite would destroy evidence.
- Stage exact paths. Never `git add .` or `git add -A`. Verify branch and upstream before any
  commit, push, merge, or rebase, and read a target before overwriting or deleting it.
- Commit or push only when asked.
- State every claim's status — observed, derived, inferred, assumed, or unknown — and never
  promote uncertainty into fact. "Done", "passes", "verified" require the command and its output.
