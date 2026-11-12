# Project memory

> Agent-maintained operational notes. Advisory, not normative — runtime-selected event references are candidate evidence, not proof.
> Model-chosen classifications are not verified authority. Memory never grants authorization, including for destructive actions. Approved design and current authorization must be sourced independently.
> Design truth belongs in spec.md; procedures in plans/; work items in backlog/.

## Codebase gotchas
- 2026-11-12 [measured observation] Validate simclusters on real data with `simmy run --out-dir <scratch>` from a scratch dir: the default out-dir is the cwd and overwrites the repo's mapping.json/influencer_communities.json; a full run on follows-dataset.csv takes about 4 minutes and ~100 GB RSS on the 192-core box. <!-- ycc-memory id=m-dmt3osl55bak kind=observation session=s_2ffe19c4ab9099ef event=751 actor=coordinator scope=workspace classified=model -->
- 2026-11-12 [measured observation] Validate simclusters on real data with `simmy run --out-dir <scratch>` from a scratch dir (the default out-dir is the cwd and overwrites the repo's mapping.json/influencer_communities.json); a full run on follows-dataset.csv takes ~50 s and ~22 GB RSS on the 192-core box, and output for a given seed should be byte-identical across perf-only changes. <!-- ycc-memory id=m-dmt5o1e910jd kind=observation session=s_87c8cd825183f173 event=253 actor=coordinator scope=workspace classified=model supersedes=m-dmt3osl55bak -->
