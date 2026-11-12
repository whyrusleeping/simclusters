package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/bluesky-social/indigo/atproto/identity"
	"github.com/bluesky-social/indigo/atproto/syntax"
	"github.com/urfave/cli/v3"
	"gorm.io/gorm"
)

var queryCmd = &cli.Command{
	Name: "query", Usage: "Interactively query user communities and influencers",
	Flags:  []cli.Flag{mappingFlag(), influencersFlag()},
	Action: runQuery,
}

type scoredInfluencer struct {
	ID    int64
	Score float64
}

// rankInfluencers indexes communities in descending score order, breaking ties by ID.
func rankInfluencers(communities assignments) map[int64][]scoredInfluencer {
	result := make(map[int64][]scoredInfluencer)
	for id, scores := range communities {
		for community, score := range scores {
			result[community] = append(result[community], scoredInfluencer{ID: id, Score: score})
		}
	}
	for _, influencers := range result {
		slices.SortFunc(influencers, func(a, b scoredInfluencer) int {
			if a.Score > b.Score {
				return -1
			}
			if a.Score < b.Score {
				return 1
			}
			if a.ID < b.ID {
				return -1
			}
			if a.ID > b.ID {
				return 1
			}
			return 0
		})
	}
	return result
}

func runQuery(ctx context.Context, cmd *cli.Command) error {
	db, err := openDatabase(ctx)
	if err != nil {
		return err
	}

	fmt.Println("Initializing identity resolver...")
	resolver := identity.DefaultDirectory()

	fmt.Printf("Loading user mappings from %s...\n", cmd.String("mapping"))
	userMappings, err := loadAssignments(cmd.String("mapping"))
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d user->community mappings\n", len(userMappings))

	fmt.Printf("Loading influencer communities from %s...\n", cmd.String("influencers"))
	influencerCommunities, err := loadAssignments(cmd.String("influencers"))
	if err != nil {
		return err
	}
	fmt.Printf("Loaded %d influencer->community mappings\n", len(influencerCommunities))

	commToInfluencers := rankInfluencers(influencerCommunities)
	fmt.Printf("Found %d unique communities\n", len(commToInfluencers))
	fmt.Printf("Found %d users\n", len(userMappings))
	fmt.Println("\nEnter DIDs or handles to query (or 'quit' to exit):")

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

		var did string
		if strings.HasPrefix(input, "did:") {
			did = input
		} else {
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

		var repo Repo
		if err := db.Table("repos").Where("did = ?", did).First(&repo).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				fmt.Printf("DID %s not found in database\n", did)
				continue
			}
			return fmt.Errorf("look up DID: %w", err)
		}
		nodeID := repo.ID
		userComms, found := userMappings[nodeID]
		if !found {
			fmt.Printf("User %d not found in user mappings\n", nodeID)
		}
		infcom, found := influencerCommunities[nodeID]
		if !found {
			fmt.Printf("User %d not found in influencer mappings\n", nodeID)
		}

		fmt.Printf("\nUser: %d (%s)\n", nodeID, did)
		fmt.Printf("Interested in %d communities:\n", len(userComms))
		// A community can appear in both the interest and influencer lists.
		reposByCommunity := make(map[int64]map[int64]Repo)
		printCommunity := func(commID int64, score float64) error {
			influencers := commToInfluencers[commID]
			fmt.Printf("Community %d (score: %.3f, %d influencers)\n", commID, score, len(influencers))
			fmt.Println("  Top influencers:")
			top := influencers[:min(10, len(influencers))]
			reposByID, cached := reposByCommunity[commID]
			if !cached && len(top) > 0 {
				reposByID = make(map[int64]Repo, len(top))
				ids := make([]int64, len(top))
				for i, inf := range top {
					ids[i] = inf.ID
				}
				var repos []Repo
				if err := db.Table("repos").Select("id, did").Where("id IN ?", ids).Find(&repos).Error; err != nil {
					return fmt.Errorf("look up community %d influencers: %w", commID, err)
				}
				for _, repo := range repos {
					reposByID[repo.ID] = repo
				}
				reposByCommunity[commID] = reposByID
			}
			for _, inf := range top {
				infRepo, found := reposByID[inf.ID]
				if !found {
					fmt.Printf("    %d (not in db, score: %.3f)\n", inf.ID, inf.Score)
					continue
				}
				infDID, err := syntax.ParseDID(infRepo.DID)
				if err == nil {
					infIdent, err := resolver.LookupDID(ctx, infDID)
					if err == nil && infIdent.Handle.String() != "" {
						fmt.Printf("    @%s (%d, score: %.3f)\n", infIdent.Handle.String(), inf.ID, inf.Score)
						continue
					}
				}
				fmt.Printf("    %d (%s, score: %.3f)\n", inf.ID, infRepo.DID, inf.Score)
			}
			if len(influencers) == 0 {
				fmt.Println("    (no influencers)")
			}
			fmt.Println()
			return nil
		}

		for commID, score := range userComms {
			if err := printCommunity(commID, score); err != nil {
				return err
			}
		}
		fmt.Printf("Influencer in %d communities:\n\n", len(infcom))
		for commID, score := range infcom {
			if err := printCommunity(commID, score); err != nil {
				return err
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading input: %w", err)
	}
	fmt.Println("Goodbye!")
	return nil
}
