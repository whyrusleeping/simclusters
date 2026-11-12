---
id: "0002"
title: 'Cut simmy run time and memory: CSR follow graph, dense similarity matrix'
status: done
priority: 2
created: "2026-11-12"
updated: "2026-11-12"
depends_on: []
spec_refs: []
---

## Description

User asked to optimize further: the full run on follows-dataset.csv took 3:10 wall and 104 GB peak RSS. Stages: ingest 94 s, influencers 7 s, similarity 38 s, simGraph 17 s, MH 19 s, output 5 s.

Approach:
- Store the follow graph as a CSR over dense user indices instead of per-user roaring64 bitmaps in both directions. Keep only the following direction and in-degrees; clustering never needs the full followers direction.
- Similarity stage uses dense arrays, not a per-user map. Its result is an upper-triangle matrix of rows (int32 column, float64 weight), so there is no 24-byte SimEdge slice and no concatenation copy.
- newSimGraph builds from the matrix in parallel with no maps; interestedIn looks up by dense index.
- The checkpoint stores users and the following CSR (format version 3); matrix.bin and influencers.bin keep their format.

Acceptance:
- On the real dataset, influencer_communities.json and mapping.json are byte-identical to the baseline run with the same flags and seed.
- Wall time and peak RSS are both substantially lower; before/after numbers are reported.
- Tests pass and cover the CSR index (dedupe, IDs at or above 2^32, adds after reads), similarity against brute force, and checkpoint round trip and corruption. gofmt and vet are clean.

## Outcome

Full run on follows-dataset.csv went from 2:59 wall / 107.6 GB peak RSS (warm cache) to 0:49 / 22.5 GB, and resume from checkpoint takes ~45 s. Fresh, checkpointed and resumed runs all produce influencer_communities.json and mapping.json byte-identical to the baseline. Checkpoint format is now v3: older checkpoints must be rebuilt from CSV. Tests, race detector, vet and staticcheck are clean; roaring and sync-map dependencies are removed.

Commit: Cut run time 2:59 -> 0:49 and peak RSS 108 GB -> 22 GB
