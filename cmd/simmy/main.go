package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
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
	// Start pprof server
	go func() {
		http.ListenAndServe(":9123", nil)
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

	mat, err := loadMatrixBin("checkpoint-big3/matrix.bin")
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d edges\n", len(mat))

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
			infcom, ok := influencerCommunities[nodeID]
			if !ok {
				fmt.Printf("User %d not found in mappings\n", nodeID)
				continue
			}
			for commID, weight := range infcom {
				userComms = map[int64]float64{commID: weight}
			}
		}

		fmt.Printf("\nUser: %d (%s)\n", nodeID, did)
		fmt.Printf("Interested in %d communities:\n\n", len(userComms))

		// Sort communities by score (descending)
		type commScore struct {
			commID int64
			score  float64
		}
		var sortedComms []commScore
		for commID, score := range userComms {
			sortedComms = append(sortedComms, commScore{commID, score})
		}
		slices.SortFunc(sortedComms, func(a, b commScore) int {
			if a.score != b.score {
				if a.score > b.score {
					return -1
				}
				return 1
			}
			return int(a.commID - b.commID)
		})

		// Show each community and its top influencers
		for _, cs := range sortedComms {
			influencers := commToInfluencers[cs.commID]
			fmt.Printf("Community %d (score: %.3f, %d influencers)\n", cs.commID, cs.score, len(influencers))

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

		fmt.Println("Influencer edges for user")
		for k, v := range mat {
			if k.A == nodeID || k.B == nodeID {
				fmt.Printf("%d -> %d: %f\n", k.A, k.B, v)
			}
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

func loadMatrixBin(path string) (map[matKey]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 256*1024)

	// Read count
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read count: %w", err)
	}

	// Read matrix entries
	matrix := make(map[matKey]float64, count)
	buf := make([]byte, 24)
	for i := int64(0); i < count; i++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, err
		}

		var a, b int64
		var val float64

		a = int64(binary.LittleEndian.Uint64(buf[:8]))
		b = int64(binary.LittleEndian.Uint64(buf[8:16]))
		valb := binary.LittleEndian.Uint64(buf[16:])
		val = math.Float64frombits(valb)

		matrix[matKey{A: a, B: b}] = val

		if (i+1)%int64(10000) == 0 {
			slog.Debug("Matrix load progress", "loaded", i+1, "total", count)
		}
	}

	return matrix, nil
}
