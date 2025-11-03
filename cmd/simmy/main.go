package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/bluesky-social/indigo/atproto/identity"
	"github.com/bluesky-social/indigo/atproto/syntax"
	"github.com/schollz/progressbar/v3"
	"github.com/urfave/cli/v3"
	"github.com/whyrusleeping/simclusters"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

func main() {
	// Enable profiling for mutex contention and blocking
	runtime.SetMutexProfileFraction(1)
	runtime.SetBlockProfileRate(1)

	// Start pprof server
	go func() {
		log := slog.New(slog.NewTextHandler(os.Stderr, nil))
		log.Info("pprof server listening", "addr", "http://localhost:9123/debug/pprof")
		if err := http.ListenAndServe(":9123", nil); err != nil {
			log.Error("pprof server failed", "error", err)
		}
	}()

	cmd := &cli.Command{
		Name:  "simmy",
		Usage: "SimClusters algorithm implementation",
		Commands: []*cli.Command{
			{
				Name:  "run",
				Usage: "Run the SimClusters algorithm",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:    "checkpoint",
						Aliases: []string{"c"},
						Usage:   "Path to save checkpoint after initializeInfluencerEdges",
					},
					&cli.StringFlag{
						Name:    "from-checkpoint",
						Aliases: []string{"f"},
						Usage:   "Path to load checkpoint and resume computation",
					},
				},
				Action: runClustering,
			},
			{
				Name:  "query",
				Usage: "Interactively query user communities and influencers",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:    "mapping",
						Aliases: []string{"m"},
						Usage:   "Path to mapping.json (user->community mappings)",
						Value:   "mapping.json",
					},
					&cli.StringFlag{
						Name:    "influencers",
						Aliases: []string{"i"},
						Usage:   "Path to influencer_communities.json",
						Value:   "influencer_communities.json",
					},
				},
				Action: runQuery,
			},
			{
				Name:  "debug-users",
				Usage: "Debug missing high-follower users in the output",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:    "checkpoint",
						Aliases: []string{"c"},
						Usage:   "Path to checkpoint directory",
						Value:   "checkpoint-big5",
					},
					&cli.StringSliceFlag{
						Name:    "users",
						Aliases: []string{"u"},
						Usage:   "User IDs to debug (can specify multiple)",
					},
				},
				Action: runDebugUsers,
			},
			{
				Name:  "export-csv",
				Usage: "Export community mappings to CSV with DIDs",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:    "mapping",
						Aliases: []string{"m"},
						Usage:   "Path to mapping.json (user->community mappings)",
						Value:   "mapping.json",
					},
					&cli.StringFlag{
						Name:    "influencers",
						Aliases: []string{"i"},
						Usage:   "Path to influencer_communities.json",
						Value:   "influencer_communities.json",
					},
					&cli.StringFlag{
						Name:    "output",
						Aliases: []string{"o"},
						Usage:   "Output CSV file path",
						Value:   "communities.csv",
					},
				},
				Action: runExportCSV,
			},
			buildDatasetCmd,
		},
	}

	if err := cmd.Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func loadEdgesFromFile(simc *simclusters.SimClusters, fn string) error {
	fi, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer fi.Close()

	rr := csv.NewReader(fi)

	var edges []simclusters.Edge
	for i := 0; true; i++ {
		rec, err := rr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}

		from, err := strconv.Atoi(rec[1])
		if err != nil {
			return err
		}

		to, err := strconv.Atoi(rec[2])
		if err != nil {
			return err
		}

		edges = append(edges, simclusters.Edge{
			From: int64(from),
			To:   int64(to),
		})
	}

	simc.IngestEdges(edges)

	return nil
}

func runClustering(ctx context.Context, cmd *cli.Command) error {
	cfg := simclusters.DefaultConfig()
	simc := simclusters.NewSimClusters(cfg)

	checkpointPath := cmd.String("checkpoint")
	fromCheckpoint := cmd.String("from-checkpoint")

	// Set checkpoint path if provided
	if checkpointPath != "" {
		simc.CheckpointPath = checkpointPath
		slog.Info("Checkpoint will be saved to", "path", checkpointPath)
	}

	// If resuming from checkpoint, skip data loading
	if fromCheckpoint != "" {
		slog.Info("Resuming computation from checkpoint", "path", fromCheckpoint)
		return simc.ComputeClustersFromCheckpoint(fromCheckpoint)
	}

	// Load data files
	files := cmd.Args().Slice()
	if len(files) == 0 {
		return fmt.Errorf("no input files provided")
	}

	bar := progressbar.Default(int64(len(files)))
	var wg sync.WaitGroup
	sema := make(chan struct{}, 40)
	for _, fn := range files {
		wg.Add(1)
		sema <- struct{}{}
		go func(file string) {
			defer func() {
				<-sema
				wg.Done()
				bar.Add(1)
			}()

			if err := loadEdgesFromFile(simc, file); err != nil {
				slog.Error("failed to load edges", "file", file, "error", err)
			}
		}(fn)
	}
	wg.Wait()

	if err := simc.ComputeClusters(); err != nil {
		return err
	}

	return nil
}

// database table for user repo lookups
type Repo struct {
	ID  int64  `gorm:"primaryKey;column:id"`
	DID string `gorm:"column:did"`
}

func runQuery(ctx context.Context, cmd *cli.Command) error {
	mappingPath := cmd.String("mapping")
	influencersPath := cmd.String("influencers")

	// Connect to database
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		return fmt.Errorf("DATABASE_URL environment variable not set")
	}

	fmt.Println("Connecting to database...")
	db, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}

	// Create identity resolver for handle lookups
	fmt.Println("Initializing identity resolver...")
	resolver := identity.DefaultDirectory()

	// Load user->community mappings (all users)
	fmt.Printf("Loading user mappings from %s...\n", mappingPath)
	mf, err := os.Open(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to open mapping file: %w", err)
	}
	defer mf.Close()

	var userMappings map[int64]map[int64]float64
	if err := json.NewDecoder(mf).Decode(&userMappings); err != nil {
		return fmt.Errorf("failed to decode mapping.json: %w", err)
	}

	fmt.Printf("Loaded %d user->community mappings\n", len(userMappings))

	// Load influencer communities
	fmt.Printf("Loading influencer communities from %s...\n", influencersPath)
	inf, err := os.Open(influencersPath)
	if err != nil {
		return fmt.Errorf("failed to open influencer file: %w", err)
	}
	defer inf.Close()

	var influencerCommunities map[int64]map[int64]float64
	if err := json.NewDecoder(inf).Decode(&influencerCommunities); err != nil {
		return fmt.Errorf("failed to decode influencer_communities.json: %w", err)
	}

	fmt.Printf("Loaded %d influencer->community mappings\n", len(influencerCommunities))

	// Build reverse index: community -> list of influencers
	commToInfluencers := make(map[int64][]int64)
	for nodeID, commSet := range influencerCommunities {
		for commID, _ := range commSet {
			commToInfluencers[commID] = append(commToInfluencers[commID], nodeID)
		}
	}

	// Sort each community's influencer list for consistent ordering
	for commID := range commToInfluencers {
		slices.Sort(commToInfluencers[commID])
	}

	fmt.Printf("Found %d unique communities\n", len(commToInfluencers))
	fmt.Printf("Found %d users\n", len(userMappings))

	fmt.Println("\nEnter DIDs or handles to query (or 'quit' to exit):")

	// Interactive REPL
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "quit" || input == "exit" || input == "q" {
			break
		}

		// Determine if input is a DID or handle
		var did string
		if strings.HasPrefix(input, "did:") {
			// Input is already a DID
			did = input
		} else {
			// Input is a handle, resolve to DID
			handleStr := strings.TrimPrefix(input, "@")
			fmt.Printf("Resolving handle %s...\n", handleStr)

			handle, err := syntax.ParseHandle(handleStr)
			if err != nil {
				fmt.Printf("Invalid handle: %v\n", err)
				continue
			}

			ident, err := resolver.LookupHandle(ctx, handle)
			if err != nil {
				fmt.Printf("Failed to resolve handle: %v\n", err)
				continue
			}
			did = ident.DID.String()
			fmt.Printf("Resolved to DID: %s\n", did)
		}

		// Look up DID in database to get node ID
		var repo Repo
		result := db.Table("repos").Where("did = ?", did).First(&repo)
		if result.Error != nil {
			if errors.Is(result.Error, gorm.ErrRecordNotFound) {
				fmt.Printf("DID %s not found in database\n", did)
			} else {
				fmt.Printf("Database error: %v\n", result.Error)
			}
			continue
		}

		nodeID := repo.ID

		// Look up user's communities
		userComms, found := userMappings[nodeID]
		if !found {
			fmt.Printf("User %d not found in user mappings\n", nodeID)
			userComms = make(map[int64]float64)
		}

		infcom, ok := influencerCommunities[nodeID]
		if !ok {
			fmt.Printf("User %d not found in influencer mappings\n", nodeID)
			infcom = make(map[int64]float64)
		}

		fmt.Printf("\nUser: %d (%s)\n", nodeID, did)
		fmt.Printf("Interested in %d communities:\n", len(userComms))

		printCommunity := func(commID int64, score float64) {
			influencers := commToInfluencers[commID]
			fmt.Printf("Community %d (score: %.3f, %d influencers)\n", commID, score, len(influencers))
			// Show top 10 influencers with their handles
			fmt.Println("  Top influencers:")
			count := 0
			for _, inf := range influencers {
				// Look up DID from database
				var infRepo Repo
				result := db.Table("repos").Where("id = ?", inf).First(&infRepo)
				if result.Error != nil {
					// Fallback to just showing node ID if not in database
					fmt.Printf("    %d (not in db)\n", inf)
				} else {
					// Resolve DID to handle
					infDID, err := syntax.ParseDID(infRepo.DID)
					if err != nil {
						fmt.Printf("    %d (%s)\n", inf, infRepo.DID)
					} else {
						infIdent, err := resolver.LookupDID(ctx, infDID)
						if err != nil || infIdent.Handle.String() == "" {
							// Fallback to DID if handle lookup fails
							fmt.Printf("    %d (%s)\n", inf, infRepo.DID)
						} else {
							// Show handle with node ID
							fmt.Printf("    @%s (%d)\n", infIdent.Handle.String(), inf)
						}
					}
				}
				count++
				if count >= 10 {
					break
				}
			}
			if len(influencers) == 0 {
				fmt.Println("    (no influencers)")
			}
			fmt.Println()

		}

		// Show each community and its top influencers
		for commID, score := range userComms {
			printCommunity(commID, score)
		}
		fmt.Printf("Influencer in %d communities:\n\n", len(infcom))
		for commID, score := range infcom {
			printCommunity(commID, score)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading input: %w", err)
	}

	fmt.Println("Goodbye!")
	return nil
}

type matKey struct {
	A, B int64
}

func runDebugUsers(ctx context.Context, cmd *cli.Command) error {
	checkpointPath := cmd.String("checkpoint")
	userIDs := cmd.StringSlice("users")

	if len(userIDs) == 0 {
		return fmt.Errorf("no user IDs specified (use -u or --users)")
	}

	// Parse user IDs
	var missingUsers []int64
	for _, userStr := range userIDs {
		userID, err := strconv.ParseInt(userStr, 10, 64)
		if err != nil {
			return fmt.Errorf("invalid user ID %q: %w", userStr, err)
		}
		missingUsers = append(missingUsers, userID)
	}

	// Load checkpoint
	fmt.Printf("Loading checkpoint from %s...\n", checkpointPath)
	state, err := simclusters.LoadCheckpoint(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to load checkpoint: %w", err)
	}

	fmt.Printf("Loaded checkpoint with %d influencers, %d edges\n",
		len(state.Influencers), len(state.InfluencerMatrix))

	// Check if they're influencers
	influencerSet := make(map[int64]bool)
	for _, id := range state.Influencers {
		influencerSet[id] = true
	}

	fmt.Println("\n=== Checking Missing Users ===")
	for _, userID := range missingUsers {
		fmt.Printf("\nUser %d:\n", userID)

		// Check if in influencer list
		if !influencerSet[userID] {
			fmt.Printf("  ❌ NOT in influencer list\n")
			continue
		}
		fmt.Printf("  ✓ In influencer list\n")

		// Check follower count
		if user := state.GraphIndex.Users[userID]; user != nil {
			fmt.Printf("  Followers: %d\n", user.FollowerCount)
		}

		// Check edge count at different thresholds
		edgeCount := 0
		edgesAbove01 := 0
		edgesAbove015 := 0
		edgesAbove02 := 0
		var minSim, maxSim float64 = 1.0, 0.0
		var sumSim float64

		for edge, weight := range state.InfluencerMatrix {
			if edge.A == userID || edge.B == userID {
				edgeCount++
				sumSim += weight
				if weight < minSim {
					minSim = weight
				}
				if weight > maxSim {
					maxSim = weight
				}
				if weight >= 0.1 {
					edgesAbove01++
				}
				if weight >= 0.15 {
					edgesAbove015++
				}
				if weight >= 0.2 {
					edgesAbove02++
				}
			}
		}

		fmt.Printf("  Total edges in checkpoint: %d\n", edgeCount)
		if edgeCount > 0 {
			fmt.Printf("  Edges >= 0.10: %d (%.1f%%)\n", edgesAbove01, 100.0*float64(edgesAbove01)/float64(edgeCount))
			fmt.Printf("  Edges >= 0.15: %d (%.1f%%)\n", edgesAbove015, 100.0*float64(edgesAbove015)/float64(edgeCount))
			fmt.Printf("  Edges >= 0.20: %d (%.1f%%)\n", edgesAbove02, 100.0*float64(edgesAbove02)/float64(edgeCount))
			fmt.Printf("  Similarity range: %.4f - %.4f (avg: %.4f)\n", minSim, maxSim, sumSim/float64(edgeCount))
		}

		// Check if in final output files
		outputData, err := os.ReadFile("influencer_communities.json")
		if err != nil {
			fmt.Printf("  ⚠ Could not read influencer_communities.json: %v\n", err)
		} else {
			var communities map[string]map[string]float64
			if err := json.Unmarshal(outputData, &communities); err != nil {
				fmt.Printf("  ⚠ Could not parse influencer_communities.json: %v\n", err)
			} else {
				userKey := fmt.Sprintf("%d", userID)
				if comms, exists := communities[userKey]; exists {
					fmt.Printf("  ✓ In influencer_communities.json with %d communities\n", len(comms))
				} else {
					fmt.Printf("  ❌ NOT in influencer_communities.json\n")
				}
			}
		}

		// Check mapping.json
		mappingData, err := os.ReadFile("mapping.json")
		if err != nil {
			fmt.Printf("  ⚠ Could not read mapping.json: %v\n", err)
		} else {
			var mapping map[string]map[string]float64
			if err := json.Unmarshal(mappingData, &mapping); err != nil {
				fmt.Printf("  ⚠ Could not parse mapping.json: %v\n", err)
			} else {
				userKey := fmt.Sprintf("%d", userID)
				if comms, exists := mapping[userKey]; exists {
					fmt.Printf("  ✓ In mapping.json with %d communities\n", len(comms))
				} else {
					fmt.Printf("  ❌ NOT in mapping.json\n")
				}
			}
		}
	}

	fmt.Println("\n=== Analysis Summary ===")
	fmt.Println("If users have edges in checkpoint but not in output:")
	fmt.Println("  1. Check if edges were filtered by 0.2 threshold (see edges >= 0.20)")
	fmt.Println("  2. Check if user ended with 0 communities after MH optimization")
	fmt.Println("  3. Check if user was isolated in graph after filtering")

	return nil
}

func runExportCSV(ctx context.Context, cmd *cli.Command) error {
	mappingPath := cmd.String("mapping")
	influencersPath := cmd.String("influencers")
	outputPath := cmd.String("output")

	// Connect to database
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		return fmt.Errorf("DATABASE_URL environment variable not set")
	}

	fmt.Println("Connecting to database...")
	db, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{})
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
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

	// Load influencer communities
	fmt.Printf("Loading influencer communities from %s...\n", influencersPath)
	inf, err := os.Open(influencersPath)
	if err != nil {
		return fmt.Errorf("failed to open influencer file: %w", err)
	}
	defer inf.Close()

	var influencerCommunities map[int64]map[int64]float64
	if err := json.NewDecoder(inf).Decode(&influencerCommunities); err != nil {
		return fmt.Errorf("failed to decode influencer_communities.json: %w", err)
	}
	fmt.Printf("Loaded %d influencer->community mappings\n", len(influencerCommunities))

	// Load user->community mappings
	fmt.Printf("Loading user mappings from %s...\n", mappingPath)
	mf, err := os.Open(mappingPath)
	if err != nil {
		return fmt.Errorf("failed to open mapping file: %w", err)
	}
	defer mf.Close()

	var userMappings map[int64]map[int64]float64
	if err := json.NewDecoder(mf).Decode(&userMappings); err != nil {
		return fmt.Errorf("failed to decode mapping.json: %w", err)
	}
	fmt.Printf("Loaded %d user->community mappings\n", len(userMappings))

	// Create output CSV
	fmt.Printf("Writing CSV to %s...\n", outputPath)
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	// Write header
	if err := writer.Write([]string{"DID", "communityID", "is_influencer", "score"}); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Track statistics
	var totalRows, missingDIDs int

	// Write influencer rows
	for uid, communities := range influencerCommunities {
		did, hasDID := uidToDID[uid]
		if !hasDID {
			missingDIDs++
			continue // Skip users without DIDs
		}

		for commID, score := range communities {
			row := []string{
				did,
				fmt.Sprintf("%d", commID),
				"true",
				fmt.Sprintf("%.6f", score),
			}
			if err := writer.Write(row); err != nil {
				return fmt.Errorf("failed to write CSV row: %w", err)
			}
			totalRows++
		}
	}

	// Write regular user rows
	for uid, communities := range userMappings {
		did, hasDID := uidToDID[uid]
		if !hasDID {
			missingDIDs++
			continue // Skip users without DIDs
		}

		for commID, score := range communities {
			row := []string{
				did,
				fmt.Sprintf("%d", commID),
				"false",
				fmt.Sprintf("%.6f", score),
			}
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
