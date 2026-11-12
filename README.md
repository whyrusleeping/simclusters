# SimClusters

A Go implementation of community detection for social graphs, after the [SimClusters paper](https://dl.acm.org/doi/10.1145/3394486.3403370). Influencers (accounts with many followers) are linked by the cosine similarity of their follower sets and grouped into overlapping communities; every user is then described by the communities of the influencers they follow.

## Build

```bash
go build ./cmd/simmy
```

## Input

Supply a follow graph as CSV with numeric IDs:

```csv
id,author,subject
1234,2345,3456
```

This says user 2345 follows user 3456 (follow ID 1234). The header row is optional. One big file is fine: it is parsed in parallel, as are multiple input files.

### Build a dataset

Build a filtered CSV from active accounts in PostgreSQL:

```bash
./simmy build-dataset --output follows-dataset.csv --author-active-days 180 --subject-active-days 60
```

`build-dataset`, `query`, `export-csv` and `cluster-posts` require the `DATABASE_URL` environment variable.

## Run clustering

```bash
./simmy run --checkpoint ./checkpoints --out-dir ./results follows-dataset.csv
./simmy run --from-checkpoint ./checkpoints --out-dir ./results
```

Main flags:

- `--checkpoint` saves the follow and similarity graphs; `--from-checkpoint` loads them instead of CSV input.
- `--out-dir` chooses where the result files are written (default: current directory).
- `--min-similarity` sets the lowest similarity used for clustering (default: 0.2). Influencers with no edge at or above it are left unclustered; the run logs how many.
- `--similarity-floor` stores similarities down to this value in the checkpoint, so a resumed run can lower `--min-similarity` without recomputing them (default: the `--min-similarity` value).
- `--max-neighbors` keeps only each influencer's N strongest similarity edges (default: 100; 0 keeps all, which on a dense graph produces a few huge communities).
- `--communities` sets the number of communities; `--epochs` sets the sampling passes.
- `--seed` sets the random seed; the same input, flags and seed reproduce the same output.
- `--workers` sets parallelism.

See `./simmy run --help` for defaults and the remaining tuning flags.

## Explore results

### Query communities

Interactively enter DIDs or handles; top influencers are ranked by their community scores:

```bash
./simmy query --mapping results/mapping.json --influencers results/influencer_communities.json
```

### Export to CSV

Writes `DID,communityID,is_influencer,score`:

```bash
./simmy export-csv --mapping results/mapping.json --influencers results/influencer_communities.json --output clusters.csv
```

### Debug missing users

```bash
./simmy debug-users --checkpoint ./checkpoints --users 2345,3456 --mapping results/mapping.json --influencers results/influencer_communities.json
```

### Find posts by cluster

Find posts similar to a cluster's centroid (the cluster ID is required):

```bash
./simmy cluster-posts --influencers results/influencer_communities.json --limit 100 --days 5 --model text-embedding-3-small --candidate-days 3 --min-likes 10 42
```

`--days` is the centroid look-back window; `--candidate-days` is the candidate window. Candidates must have **more than** `--min-likes` likes.

## Output

Both result files are JSON objects of the form `{"<user id>": {"<community id>": <score>, ...}, ...}`:

- `influencer_communities.json`: scores are the share (0–1] of an influencer's similarity weight going to members of each community. Higher means more central to that community.
- `mapping.json`: scores are a user's normalized interest in each of their top communities.

## Profiling

Profiling is off by default. Opt in with the root flag `--pprof <addr>`, for example `./simmy --pprof localhost:9123 query`. This serves `/debug/pprof/` and enables mutex and block profiles. Only use a trusted bind address; the endpoint is unauthenticated.
