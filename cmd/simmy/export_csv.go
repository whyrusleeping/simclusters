package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"

	"github.com/urfave/cli/v3"
)

var exportCSVCmd = &cli.Command{
	Name: "export-csv", Usage: "Export community mappings to CSV with DIDs",
	Flags: []cli.Flag{
		mappingFlag(), influencersFlag(),
		&cli.StringFlag{Name: "output", Aliases: []string{"o"}, Usage: "Output CSV file path", Value: "communities.csv"},
	},
	Action: runExportCSV,
}

func runExportCSV(ctx context.Context, cmd *cli.Command) error {
	mappingPath := cmd.String("mapping")
	influencersPath := cmd.String("influencers")
	outputPath := cmd.String("output")

	db, err := openDatabase(ctx)
	if err != nil {
		return err
	}

	// Load ALL DID<>UID mappings upfront
	fmt.Println("Loading all DID<>UID mappings from database...")
	var repos []Repo
	if err := db.Table("repos").Select("id, did").Find(&repos).Error; err != nil {
		return fmt.Errorf("failed to load repos: %w", err)
	}

	uidToDID := make(map[int64]string)
	for _, repo := range repos {
		uidToDID[repo.ID] = repo.DID
	}
	fmt.Printf("Loaded %d DID mappings\n", len(uidToDID))

	fmt.Printf("Loading influencer communities from %s...\n", influencersPath)
	influencerCommunities, err := loadAssignments(influencersPath)
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d influencer->community mappings\n", len(influencerCommunities))

	fmt.Printf("Loading user mappings from %s...\n", mappingPath)
	userMappings, err := loadAssignments(mappingPath)
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d user->community mappings\n", len(userMappings))

	fmt.Printf("Writing CSV to %s...\n", outputPath)
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()
	if err := writer.Write([]string{"DID", "communityID", "is_influencer", "score"}); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	var totalRows, missingDIDs int
	for uid, communities := range influencerCommunities {
		did, hasDID := uidToDID[uid]
		if !hasDID {
			missingDIDs++
			continue
		}
		for commID, score := range communities {
			row := []string{did, fmt.Sprintf("%d", commID), "true", fmt.Sprintf("%.6f", score)}
			if err := writer.Write(row); err != nil {
				return fmt.Errorf("failed to write CSV row: %w", err)
			}
			totalRows++
		}
	}

	for uid, communities := range userMappings {
		did, hasDID := uidToDID[uid]
		if !hasDID {
			missingDIDs++
			continue
		}
		for commID, score := range communities {
			row := []string{did, fmt.Sprintf("%d", commID), "false", fmt.Sprintf("%.6f", score)}
			if err := writer.Write(row); err != nil {
				return fmt.Errorf("failed to write CSV row: %w", err)
			}
			totalRows++
		}
	}

	writer.Flush()
	if err := writer.Error(); err != nil {
		return fmt.Errorf("failed to flush CSV: %w", err)
	}
	fmt.Printf("\n✓ Export complete!\n")
	fmt.Printf("  Total rows written: %d\n", totalRows)
	fmt.Printf("  Users without DIDs: %d\n", missingDIDs)
	fmt.Printf("  Output file: %s\n", outputPath)
	return nil
}
