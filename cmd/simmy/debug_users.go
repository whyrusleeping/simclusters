package main

import (
	"context"
	"fmt"
	"strconv"

	"github.com/urfave/cli/v3"
	"github.com/whyrusleeping/simclusters"
)

var debugUsersCmd = &cli.Command{
	Name: "debug-users", Usage: "Debug missing high-follower users in the output",
	Flags: []cli.Flag{
		&cli.StringFlag{Name: "checkpoint", Aliases: []string{"c"}, Usage: "Path to checkpoint directory", Required: true},
		&cli.StringSliceFlag{Name: "users", Aliases: []string{"u"}, Usage: "User IDs to debug (can specify multiple)"},
		mappingFlag(), influencersFlag(),
	},
	Action: runDebugUsers,
}

type userEdgeStats struct {
	Count, Above01, Above015, Above02 int
	Min, Max, Sum                     float64
}

// collectUserEdgeStats visits the edges only once, counting each requested endpoint.
func collectUserEdgeStats(users []int64, edges []simclusters.SimEdge) map[int64]userEdgeStats {
	stats := make(map[int64]userEdgeStats, len(users))
	for _, id := range users {
		stats[id] = userEdgeStats{}
	}
	for _, edge := range edges {
		for i, id := range [2]int64{edge.A, edge.B} {
			if i == 1 && edge.A == edge.B {
				continue
			}
			s, requested := stats[id]
			if !requested {
				continue
			}
			if s.Count == 0 {
				s.Min, s.Max = edge.Weight, edge.Weight
			} else {
				s.Min = min(s.Min, edge.Weight)
				s.Max = max(s.Max, edge.Weight)
			}
			s.Count++
			s.Sum += edge.Weight
			if edge.Weight >= 0.1 {
				s.Above01++
			}
			if edge.Weight >= 0.15 {
				s.Above015++
			}
			if edge.Weight >= 0.2 {
				s.Above02++
			}
			stats[id] = s
		}
	}
	return stats
}

func runDebugUsers(ctx context.Context, cmd *cli.Command) error {
	checkpointPath := cmd.String("checkpoint")
	userIDs := cmd.StringSlice("users")
	if len(userIDs) == 0 {
		return fmt.Errorf("no user IDs specified (use -u or --users)")
	}
	var missingUsers []int64
	for _, userStr := range userIDs {
		userID, err := strconv.ParseInt(userStr, 10, 64)
		if err != nil {
			return fmt.Errorf("invalid user ID %q: %w", userStr, err)
		}
		missingUsers = append(missingUsers, userID)
	}

	fmt.Printf("Loading checkpoint from %s...\n", checkpointPath)
	state, err := simclusters.LoadCheckpoint(checkpointPath, simclusters.LoadOptions{Followers: true})
	if err != nil {
		return fmt.Errorf("failed to load checkpoint: %w", err)
	}
	fmt.Printf("Loaded checkpoint with %d influencers, %d edges\n", len(state.Influencers), len(state.Edges))

	influencerCommunities, err := loadAssignments(cmd.String("influencers"))
	if err != nil {
		return err
	}
	userMappings, err := loadAssignments(cmd.String("mapping"))
	if err != nil {
		return err
	}
	stats := collectUserEdgeStats(missingUsers, state.Edges)
	influencerSet := make(map[int64]bool)
	for _, id := range state.Influencers {
		influencerSet[id] = true
	}

	fmt.Println("\n=== Checking Missing Users ===")
	for _, userID := range missingUsers {
		fmt.Printf("\nUser %d:\n", userID)
		if !influencerSet[userID] {
			fmt.Printf("  ❌ NOT in influencer list\n")
			continue
		}
		fmt.Printf("  ✓ In influencer list\n")
		fmt.Printf("  Followers: %d\n", state.GraphIndex.FollowerCount(userID))

		s := stats[userID]
		fmt.Printf("  Total edges in checkpoint: %d\n", s.Count)
		if s.Count > 0 {
			fmt.Printf("  Edges >= 0.10: %d (%.1f%%)\n", s.Above01, 100.0*float64(s.Above01)/float64(s.Count))
			fmt.Printf("  Edges >= 0.15: %d (%.1f%%)\n", s.Above015, 100.0*float64(s.Above015)/float64(s.Count))
			fmt.Printf("  Edges >= 0.20: %d (%.1f%%)\n", s.Above02, 100.0*float64(s.Above02)/float64(s.Count))
			fmt.Printf("  Similarity range: %.4f - %.4f (avg: %.4f)\n", s.Min, s.Max, s.Sum/float64(s.Count))
		}

		if comms, exists := influencerCommunities[userID]; exists {
			fmt.Printf("  ✓ In %s with %d communities\n", cmd.String("influencers"), len(comms))
		} else {
			fmt.Printf("  ❌ NOT in %s\n", cmd.String("influencers"))
		}
		if comms, exists := userMappings[userID]; exists {
			fmt.Printf("  ✓ In %s with %d communities\n", cmd.String("mapping"), len(comms))
		} else {
			fmt.Printf("  ❌ NOT in %s\n", cmd.String("mapping"))
		}
	}

	fmt.Println("\n=== Analysis Summary ===")
	fmt.Println("Edges below run --min-similarity (default 0.2) are not used for clustering.")
	fmt.Println("Influencers with no edge at or above that threshold are left unclustered; run logs how many.")
	return nil
}
