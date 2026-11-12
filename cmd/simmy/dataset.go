package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/urfave/cli/v3"
)

var buildDatasetCmd = &cli.Command{
	Name:  "build-dataset",
	Usage: "Build a filtered follow dataset for SimClusters training",
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:    "output",
			Aliases: []string{"o"},
			Usage:   "Output CSV file path",
			Value:   "follows-dataset.csv",
		},
		&cli.IntFlag{
			Name:  "author-active-days",
			Usage: "Number of days to look back for author activity (likes)",
			Value: 180, // 6 months
		},
		&cli.IntFlag{
			Name:  "subject-active-days",
			Usage: "Number of days to look back for subject activity (posts)",
			Value: 60, // 2 months
		},
	},
	Action: runBuildDataset,
}

func runBuildDataset(ctx context.Context, cmd *cli.Command) error {
	outputPath := cmd.String("output")
	authorActiveDays := cmd.Int("author-active-days")
	subjectActiveDays := cmd.Int("subject-active-days")

	db, err := openDatabase(ctx)
	if err != nil {
		return err
	}

	// Calculate cutoff timestamps
	authorCutoff := time.Now().AddDate(0, 0, -authorActiveDays)
	subjectCutoff := time.Now().AddDate(0, 0, -subjectActiveDays)

	fmt.Printf("Filtering follows:\n")
	fmt.Printf("  - Authors with likes since %s (%d days)\n", authorCutoff.Format("2006-01-02"), authorActiveDays)
	fmt.Printf("  - Subjects with posts since %s (%d days)\n", subjectCutoff.Format("2006-01-02"), subjectActiveDays)
	fmt.Println()

	// Build SQL query with CTEs for efficient filtering
	query := `
	WITH active_likers AS (
		-- Authors who have created at least one like in the past N months
		SELECT DISTINCT author
		FROM likes
		WHERE created >= ?
	),
	active_posters AS (
		-- Subjects who have posted at least once in the past N months
		SELECT DISTINCT author
		FROM posts
		WHERE created >= ?
	)
	-- Select follows where both author and subject are active
	SELECT
		f.id,
		f.author,
		f.subject
	FROM follows f
	INNER JOIN active_likers al ON f.author = al.author
	INNER JOIN active_posters ap ON f.subject = ap.author
	`

	fmt.Println("Executing query to find filtered follows...")

	// Execute main query
	rows, err := db.Raw(query, authorCutoff, subjectCutoff).Rows()
	if err != nil {
		return fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	// Create output CSV file
	fmt.Printf("Writing to %s...\n", outputPath)
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"id", "author", "subject"}); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Process and write rows with progress bar
	bar := progressbar.Default(-1)
	var count int64

	for rows.Next() {
		var followID int64
		var author int64
		var subject int64

		if err := rows.Scan(&followID, &author, &subject); err != nil {
			return fmt.Errorf("failed to scan row: %w", err)
		}

		if err := writer.Write([]string{
			fmt.Sprintf("%d", followID),
			fmt.Sprintf("%d", author),
			fmt.Sprintf("%d", subject),
		}); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}

		count++
		bar.Add(1)
	}

	if err := rows.Err(); err != nil {
		return fmt.Errorf("error during row iteration: %w", err)
	}

	bar.Finish()

	fmt.Printf("\n✓ Dataset build complete!\n")
	fmt.Printf("  Total follows written: %d\n", count)
	fmt.Printf("  Output file: %s\n", outputPath)

	return nil
}
