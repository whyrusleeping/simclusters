// Package simclusters detects overlapping communities in a follow graph,
// following the SimClusters paper (KDD 2020):
//
//  1. Influencers (accounts with many followers) are connected by the cosine
//     similarity of their follower sets.
//  2. Neighborhood-aware Metropolis-Hastings sampling assigns each influencer
//     to a few communities of that similarity graph.
//  3. Every user is described by the communities of the influencers they
//     follow.
package simclusters

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	"github.com/schollz/progressbar/v3"
)

// Output file names, written to Options.OutputDir.
const (
	InfluencerCommunitiesFile = "influencer_communities.json"
	MappingFile               = "mapping.json"
)

// Options configures a clustering run. Start from DefaultOptions.
type Options struct {
	// MinFollowers is the smallest follower count that makes a user an
	// influencer. The cut is raised to the 90th percentile of follower
	// counts if that is higher.
	MinFollowers int

	// MinSimilarity is the lowest cosine similarity between two
	// influencers' follower sets that counts as an edge during community
	// detection.
	MinSimilarity float64
	// SimilarityFloor is the lowest similarity kept when the similarity
	// graph is computed, and so the lowest stored in a checkpoint. A resumed
	// run can use any MinSimilarity at or above it. Zero means MinSimilarity.
	SimilarityFloor float64
	// MaxNeighbors, if positive, sparsifies the similarity graph before
	// community detection: an edge is kept only if it is among the
	// MaxNeighbors strongest edges of one of its two influencers. Without
	// it a dense similarity graph collapses into a few huge communities.
	MaxNeighbors int

	NumCommunities        int     // communities to detect
	MaxCommunitiesPerNode int     // most communities one influencer can belong to (1-7)
	MaxCandidates         int     // communities weighed per influencer per step: its own plus the best of its neighbours' (up to 8)
	Alpha                 float64 // weight of similar pairs sharing a community vs dissimilar pairs sharing one
	Epsilon               float64 // chance per step of also weighing a random community
	Temperature           float64 // sampling temperature; lower is greedier
	Epochs                int     // sampling passes over all influencers
	GreedyEpochs          int     // deterministic improvement passes run afterwards

	// TopInterests is the number of communities kept per user.
	TopInterests int

	// Seed makes runs reproducible: the same input, options and seed give
	// the same output regardless of Workers.
	Seed    uint64
	Workers int // parallelism; defaults to the number of CPUs

	// CheckpointPath, if set, is a directory that receives the follow graph
	// and similarity graph once the similarity stage finishes.
	CheckpointPath string
	// OutputDir receives InfluencerCommunitiesFile and MappingFile.
	OutputDir string
	// Progress enables progress bars on stderr.
	Progress bool
}

// DefaultOptions returns the default configuration.
func DefaultOptions() Options {
	return Options{
		MinFollowers:          1300,
		MinSimilarity:         0.2,
		MaxNeighbors:          100,
		NumCommunities:        15_000,
		MaxCommunitiesPerNode: 4,
		MaxCandidates:         7,
		Alpha:                 10,
		Epsilon:               0.08,
		Temperature:           1,
		Epochs:                100,
		GreedyEpochs:          2,
		TopInterests:          5,
		Seed:                  1,
		Workers:               runtime.NumCPU(),
		OutputDir:             ".",
	}
}

func (o *Options) validate() error {
	for _, v := range []float64{o.MinSimilarity, o.SimilarityFloor, o.Alpha, o.Epsilon, o.Temperature} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return errors.New("options must be finite numbers")
		}
	}
	if o.SimilarityFloor == 0 {
		o.SimilarityFloor = o.MinSimilarity
	}
	switch {
	case o.MinFollowers < 1:
		return errors.New("MinFollowers must be at least 1")
	case o.MinSimilarity <= 0 || o.MinSimilarity > 1:
		return errors.New("MinSimilarity must be in (0, 1]")
	case o.SimilarityFloor < 0 || o.SimilarityFloor > o.MinSimilarity:
		return fmt.Errorf("SimilarityFloor must be in (0, MinSimilarity=%g]", o.MinSimilarity)
	case o.MaxNeighbors < 0:
		return errors.New("MaxNeighbors cannot be negative")
	case o.NumCommunities < 1:
		return errors.New("NumCommunities must be at least 1")
	case o.MaxCommunitiesPerNode < 1 || o.MaxCommunitiesPerNode >= maxCandidateLimit:
		return fmt.Errorf("MaxCommunitiesPerNode must be in [1, %d]", maxCandidateLimit-1)
	case o.MaxCandidates <= o.MaxCommunitiesPerNode || o.MaxCandidates > maxCandidateLimit:
		// A node in MaxCommunitiesPerNode communities needs room to weigh
		// at least one other.
		return fmt.Errorf("MaxCandidates must be in (MaxCommunitiesPerNode, %d]", maxCandidateLimit)
	case o.Alpha <= 0:
		return errors.New("alpha must be positive")
	case o.Epsilon < 0 || o.Epsilon > 1:
		return errors.New("epsilon must be in [0, 1]")
	case o.Temperature <= 0:
		return errors.New("temperature must be positive")
	case o.Epochs < 0 || o.GreedyEpochs < 0:
		return errors.New("epochs and greedy epochs cannot be negative")
	case o.TopInterests < 1:
		return errors.New("TopInterests must be at least 1")
	}
	if o.Workers < 1 {
		o.Workers = runtime.NumCPU()
	}
	if o.OutputDir == "" {
		o.OutputDir = "."
	}
	return nil
}

// SimClusters runs the clustering pipeline over a follow graph.
type SimClusters struct {
	opts  Options
	graph *GraphIndex
}

// New creates a pipeline with an empty follow graph.
func New(opts Options) (*SimClusters, error) {
	if err := opts.validate(); err != nil {
		return nil, fmt.Errorf("invalid options: %w", err)
	}
	return &SimClusters{opts: opts, graph: NewGraphIndex()}, nil
}

// Graph returns the follow graph, to be filled before ComputeClusters.
func (sc *SimClusters) Graph() *GraphIndex { return sc.graph }

// ComputeClusters runs the whole pipeline on the ingested follow graph and
// writes the result files.
func (sc *SimClusters) ComputeClusters() error {
	influencers := sc.identifyInfluencers()
	if len(influencers) == 0 {
		return errors.New("no influencers found: the follow graph is empty or no user has enough followers")
	}

	bar := sc.progressBar(len(influencers), "similarity")
	edges := buildSimilarityEdges(sc.graph, influencers, sc.opts.SimilarityFloor, sc.opts.Workers,
		func(done int) { bar.Add(done) })
	bar.Finish()

	if sc.opts.CheckpointPath != "" {
		state := &CheckpointState{
			Influencers:     influencers,
			Edges:           edges,
			SimilarityFloor: sc.opts.SimilarityFloor,
			GraphIndex:      sc.graph,
		}
		if err := SaveCheckpoint(sc.opts.CheckpointPath, state); err != nil {
			return fmt.Errorf("saving checkpoint: %w", err)
		}
	}
	return sc.cluster(influencers, edges)
}

// ComputeClustersFromCheckpoint skips ingestion and the similarity stage by
// loading their results from a checkpoint written by an earlier run.
func (sc *SimClusters) ComputeClustersFromCheckpoint(path string) error {
	state, err := LoadCheckpoint(path, LoadOptions{})
	if err != nil {
		return fmt.Errorf("loading checkpoint: %w", err)
	}
	if state.SimilarityFloor > sc.opts.MinSimilarity {
		return fmt.Errorf("checkpoint only holds similarities >= %g, below MinSimilarity %g",
			state.SimilarityFloor, sc.opts.MinSimilarity)
	}
	sc.graph = state.GraphIndex
	return sc.cluster(state.Influencers, state.Edges)
}

// identifyInfluencers returns the users whose follower count reaches both
// MinFollowers and the 90th percentile over all users, sorted by id.
func (sc *SimClusters) identifyInfluencers() []int64 {
	type user struct {
		id        int64
		followers int
	}
	var followed []user
	sc.graph.Followers.Range(func(id int64, bm *roaring64.Bitmap) bool {
		followed = append(followed, user{id, int(bm.GetCardinality())})
		return true
	})
	// Users that follow others but have no followers themselves still count
	// towards the percentile.
	unfollowed := 0
	sc.graph.Following.Range(func(id int64, _ *roaring64.Bitmap) bool {
		if _, ok := sc.graph.Followers.Load(id); !ok {
			unfollowed++
		}
		return true
	})
	if len(followed) == 0 {
		return nil
	}

	counts := make([]int, len(followed))
	for i, u := range followed {
		counts[i] = u.followers
	}
	slices.Sort(counts)
	numUsers := len(followed) + unfollowed
	p90 := 0
	if i := numUsers*9/10 - unfollowed; i >= 0 {
		p90 = counts[i]
	}
	threshold := max(p90, sc.opts.MinFollowers)

	var influencers []int64
	for _, u := range followed {
		if u.followers >= threshold {
			influencers = append(influencers, u.id)
		}
	}
	slices.Sort(influencers)
	slog.Info("influencers identified",
		"users", numUsers, "influencers", len(influencers),
		"follower_threshold", threshold, "p90_followers", p90)
	return influencers
}

// cluster runs community detection on the similarity graph and writes the
// influencer and user community files.
func (sc *SimClusters) cluster(influencers []int64, edges []SimEdge) error {
	o := &sc.opts
	g := newSimGraph(edges, o.MinSimilarity, o.MaxNeighbors, o.Workers)
	n := g.numNodes()
	if n == 0 {
		return fmt.Errorf("no influencer pair has similarity >= %g; nothing to cluster", o.MinSimilarity)
	}
	slog.Info("similarity graph ready",
		"nodes", n, "edges", g.numEdges, "min_similarity", o.MinSimilarity, "max_neighbors", o.MaxNeighbors,
		"avg_degree", fmt.Sprintf("%.1f", 2*float64(g.numEdges)/float64(n)))
	sc.logIsolated(influencers, g)

	state := detectCommunities(g, mhParams{
		numCommunities: o.NumCommunities,
		maxPerNode:     o.MaxCommunitiesPerNode,
		maxCandidates:  o.MaxCandidates,
		alpha:          o.Alpha,
		epsilon:        o.Epsilon,
		temperature:    o.Temperature,
		seed:           o.Seed,
		workers:        o.Workers,
	}, o.Epochs, o.GreedyEpochs)
	state.logCommunitySizes()
	knownFor := state.assignments()

	infPath := filepath.Join(o.OutputDir, InfluencerCommunitiesFile)
	if err := writeJSONFile(infPath, knownFor); err != nil {
		return err
	}
	slog.Info("influencer communities written", "path", infPath, "influencers", len(knownFor))

	start := time.Now()
	mapPath := filepath.Join(o.OutputDir, MappingFile)
	var users int
	err := writeFileAtomic(mapPath, func(w io.Writer) (err error) {
		users, err = writeInterestedIn(w, sc.graph, knownFor, o.TopInterests, o.Workers)
		return err
	})
	if err != nil {
		return err
	}
	slog.Info("user communities written", "path", mapPath, "users", users,
		"took", time.Since(start).Round(time.Millisecond))
	return nil
}

// logIsolated reports influencers that are too dissimilar from every other
// influencer to be part of the similarity graph. They get no community.
func (sc *SimClusters) logIsolated(influencers []int64, g *simGraph) {
	var isolated []int64
	for _, id := range influencers {
		if _, ok := slices.BinarySearch(g.ids, id); !ok {
			isolated = append(isolated, id)
		}
	}
	if len(isolated) == 0 {
		return
	}
	slog.Warn("influencers without any similarity edge are left unclustered",
		"count", len(isolated), "of", len(influencers),
		"min_similarity", sc.opts.MinSimilarity, "sample", isolated[:min(len(isolated), 10)])
}

func (sc *SimClusters) progressBar(n int, description string) *progressbar.ProgressBar {
	if !sc.opts.Progress {
		return progressbar.DefaultSilent(int64(n))
	}
	return progressbar.Default(int64(n), description)
}
