package main

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/urfave/cli/v3"
)

var clusterPostsCmd = &cli.Command{
	Name: "cluster-posts", Usage: "Find posts similar to a cluster's centroid embedding",
	ArgsUsage: "<cluster-id>",
	Flags: []cli.Flag{
		&cli.IntFlag{Name: "limit", Usage: "Number of similar posts to return", Value: 100},
		&cli.IntFlag{Name: "days", Usage: "Number of days to look back for cluster posts", Value: 5},
		&cli.StringFlag{Name: "model", Usage: "Embedding model to filter by", Value: "text-embedding-3-small"},
		&cli.IntFlag{Name: "candidate-days", Usage: "Number of days to look back for candidate posts", Value: 3},
		&cli.IntFlag{Name: "min-likes", Usage: "Candidate posts must have more than this many likes", Value: 10},
		influencersFlag(),
	},
	Action: runClusterPosts,
}

func runClusterPosts(ctx context.Context, cmd *cli.Command) error {
	args := cmd.Args().Slice()
	if len(args) == 0 {
		return fmt.Errorf("cluster ID required as argument")
	}
	clusterID, err := strconv.ParseInt(args[0], 10, 64)
	if err != nil {
		return fmt.Errorf("invalid cluster ID: %w", err)
	}

	limit := cmd.Int("limit")
	days := cmd.Int("days")
	model := cmd.String("model")
	influencersPath := cmd.String("influencers")
	db, err := openDatabase(ctx)
	if err != nil {
		return err
	}

	fmt.Printf("Loading influencer communities from %s...\n", influencersPath)
	influencerCommunities, err := loadAssignments(influencersPath)
	if err != nil {
		return err
	}

	var clusterInfluencers []int64
	for influencerID, communities := range influencerCommunities {
		if _, inCluster := communities[clusterID]; inCluster {
			clusterInfluencers = append(clusterInfluencers, influencerID)
		}
	}
	if len(clusterInfluencers) == 0 {
		return fmt.Errorf("cluster %d not found or has no influencers", clusterID)
	}
	fmt.Printf("Found %d influencers in cluster %d\n", len(clusterInfluencers), clusterID)

	// Compute the centroid and cosine distances directly in PostgreSQL with pgvector.
	fmt.Println("Computing cluster centroid and finding similar posts...")
	query := `
		WITH cluster_posts AS (
			-- Get posts from cluster influencers in the time window
			SELECT pe.embedding
			FROM posts p
			JOIN post_embeddings pe ON p.id = pe.post
			WHERE p.author = ANY($1)
			  AND p.reply_to = 0
			  AND p.deleted_at IS NULL
			  AND p.created >= NOW() - ($4::int * INTERVAL '1 day')
			  AND pe.model = $5
		),
		centroid AS (
			-- Compute the average embedding (centroid)
			SELECT AVG(embedding) as avg_embedding
			FROM cluster_posts
		)
		-- Find posts most similar to the centroid using cosine distance
		SELECT
			p.id,
			p.author,
			p.rkey,
			p.created,
			(pe.embedding <=> (SELECT avg_embedding FROM centroid)) as distance
		FROM posts p
		JOIN post_embeddings pe ON p.id = pe.post
		WHERE p.deleted_at IS NULL AND p.created > $3
		AND p.reply_to = 0
		AND pe.model = $5
		AND (select likes from post_counts where post_counts.post = p.id) > $6
		ORDER BY pe.embedding <=> (SELECT avg_embedding FROM centroid)
		LIMIT $2
	`

	type postResult struct {
		ID       int64
		Author   int64
		Rkey     string
		Created  string
		Distance float64
	}
	oldestPost := time.Now().Add(time.Hour * 24 * -time.Duration(cmd.Int("candidate-days")))
	var results []postResult
	if err := db.Raw(query, clusterInfluencers, limit, oldestPost, days, model, cmd.Int("min-likes")).Scan(&results).Error; err != nil {
		return fmt.Errorf("failed to query similar posts: %w", err)
	}
	if len(results) == 0 {
		return fmt.Errorf("no similar posts found (cluster may have no posts with embeddings in the past %d days)", days)
	}
	fmt.Printf("Found %d similar posts\n", len(results))

	authorIDs := make([]int64, len(results))
	for i, result := range results {
		authorIDs[i] = result.Author
	}
	var repos []Repo
	if err := db.Table("repos").Where("id IN ?", authorIDs).Find(&repos).Error; err != nil {
		return fmt.Errorf("failed to query repos: %w", err)
	}
	idToDID := make(map[int64]string)
	for _, repo := range repos {
		idToDID[repo.ID] = repo.DID
	}

	fmt.Printf("\nTop %d posts similar to cluster %d centroid:\n\n", len(results), clusterID)
	printed := 0
	for _, result := range results {
		if _, inCluster := influencerCommunities[result.Author][clusterID]; inCluster {
			continue
		}
		did := idToDID[result.Author]
		if did == "" {
			did = fmt.Sprintf("unknown-user-%d", result.Author)
		}
		uri := fmt.Sprintf("https://bsky.app/profile/%s/post/%s", did, result.Rkey)
		similarity := 1.0 - result.Distance
		printed++
		fmt.Printf("%3d. [%.4f] %s\n", printed, similarity, uri)
	}
	return nil
}
