# SimClusters

A Go implementation of community detection for social graphs. Identifies clusters of similar users based on their follow relationships, with a focus on grouping users around influencers.

## Overview

SimClusters analyzes a social network's follow graph to:

- Detect influencers (high-follower accounts)
- Group users into communities based on who they follow
- Use MinHash sketches for efficient similarity computation

## Installation

```bash
go build ./cmd/simmy
```

## Usage

To start, you will need a follow graph in the following csv format:

```
id,author,subject
1234,2345,3456
```

Which says the follow relation with the ID 1234 is from user 2345 to user 3456.
The program works best when this data is split up across multiple files so it
can be processed in parallel. The ID field is not super important for the
actual clustering but does help for deduplicating new follows added to the
dataset later.

### Run clustering

```bash
./simmy run --checkpoint ./checkpoints data/*.csv
```

Resume from a previous checkpoint:

```bash
./simmy run --from-checkpoint ./checkpoints
```

### Query results

Interactively explore user communities:

```bash
./simmy query --mapping mapping.json --influencers influencer_communities.json
```

### Export to CSV

```bash
./simmy export-csv --mapping mapping.json --output clusters.csv
```

### Find posts by cluster

Find posts similar to cluster centroids:

```bash
./simmy cluster-posts --limit 100 --days 5
```

## Output

The clustering produces two main files:

- `mapping.json` - Maps users to their cluster memberships
- `influencer_communities.json` - Maps influencers to their communities
