# Deliverable Audit — 2026-07-21

Baseline commit: `90725a56f28c6a5a09c0a93a31afcb15f3dfa504`

## Completeness checks

- Required numbered deliverables present and non-empty: 11/11.
- Normative atomic requirements: 51 across all 13 requested ID categories.
- Duplicate requirement IDs: 0.
- Requirement rows without a normative MUST/SHOULD term: 0.
- CR-01 through CR-15 present in the traceability matrix: 15/15.
- Project-13 major workstreams present in the traceability matrix: 9/9.
- Owner decisions present in the traceability matrix: 11/11.
- Architecture decisions AD-01 through AD-12 defined and mapped: 12/12.
- Required spikes SP-01 through SP-14 specified: 14/14.
- Evidence vocabulary includes Verified locally, Verified upstream, Inferred, Unknown, and Contradicted.
- Product capability vocabulary includes Verified, Experimental, and Unsupported.
- Local Markdown links in the authored documents resolve to existing evidence files.

## Scope audit

`git status --short` at final audit shows only three untracked research trees:

```text
?? .scratch/projects/12-structured-agents-v2-library-study/
?? .scratch/projects/13-xgrammar-and-batching-todo/
?? .scratch/projects/14-specialist-runtime-architecture-research/
```

Projects 12 and 13 predated this session and were not modified intentionally. Project 14 is the only new tree. There are no tracked source/test/deployment/dependency/lock/CI/README changes.

## Baseline command status carried into the plan

- `pytest`: 32 passed, 1 skipped.
- Ruff lint: pass.
- `ty`: pass.
- Ruff format check: fail, eight existing files would be reformatted.
- Wheel build/content: pass/clean.
- Sdist build: succeeds but content is contaminated; retained as Phase 0 release blocker CR-08.
- Focused real-object probe: all six expected finding reproductions observed, including real queue registration failure and cross-thread allowlist leakage.

The planning documents do not convert a failing baseline gate into a passing claim. They require format and sdist cleanup in the later implementation session.
