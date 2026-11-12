---
id: "0001"
title: 'Fix review findings: clustering correctness, MH performance, ingest/checkpoint robustness, CLI hygiene'
status: done
priority: 1
created: "2026-11-12"
updated: "2026-11-12"
depends_on: []
spec_refs: []
---

## Description
User accepted all findings from the code review ("fix all of them").

Acceptance:
- Clustering: per-node community cap enforced at all times; no optimizer work silently discarded; neighbour term of the objective counts a neighbour once (paper Eq. 2), no self-count; fresh and resumed runs share one code path and one parameter set; isolated influencers are logged.
- Similarity stage: MinHash prefilter uses a size-aware Jaccard bound derived from the cosine threshold; verified against brute force.
- MH kernel: dense ids/slices, uses all cores, deterministic for a given seed; real F(Z) logged.
- Ingest: header row accepted, load errors abort the run, concurrent ingest loses no edges, follower counts are unique followers, large single file parsed in parallel.
- Checkpoint: atomic save, validated load, resume does not load data it does not use.
- CLI: pprof opt-in, output dir configurable, debug-users loads JSON once, cluster-posts honours --model, query ranks influencers by score.
- Hygiene: dead code and unused deps removed, gofmt/vet/staticcheck clean, tests cover kernel, similarity, ingest, checkpoint; README matches the CLI; no credential in export-follows.sh.

## Acceptance criteria

## Work log
