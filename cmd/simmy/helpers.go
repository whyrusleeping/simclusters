package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/urfave/cli/v3"
	"github.com/whyrusleeping/simclusters"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

// Repo is the database row used for user DID lookups.
type Repo struct {
	ID  int64  `gorm:"primaryKey;column:id"`
	DID string `gorm:"column:did"`
}

type assignments map[int64]map[int64]float64

func openDatabase(ctx context.Context) (*gorm.DB, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		return nil, fmt.Errorf("DATABASE_URL environment variable not set")
	}
	fmt.Println("Connecting to database...")
	db, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	return db.WithContext(ctx), nil
}

func loadAssignments(path string) (assignments, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open assignments %q: %w", path, err)
	}
	defer f.Close()

	var result assignments
	if err := json.NewDecoder(f).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode assignments %q: %w", path, err)
	}
	return result, nil
}

func mappingFlag() cli.Flag {
	return &cli.StringFlag{
		Name: "mapping", Aliases: []string{"m"},
		Usage: "Path to mapping.json (user->community mappings)", Value: simclusters.MappingFile,
	}
}

func influencersFlag() cli.Flag {
	return &cli.StringFlag{
		Name: "influencers", Aliases: []string{"i"},
		Usage: "Path to influencer_communities.json", Value: simclusters.InfluencerCommunitiesFile,
	}
}
