package simclusters

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	"github.com/schollz/progressbar/v3"
)

// SimClusters is the main structure for the SimClusters algorithm
type SimClusters struct {
	config *Config
	mu     sync.RWMutex

	graphIndex *GraphIndex

	influencerMatrix map[matKey]float64

	MaxConcurrency int

	// CheckpointPath is the optional path to save/load checkpoints
	CheckpointPath string
}

// NewSimClusters creates a new SimClusters instance
func NewSimClusters(config *Config) *SimClusters {
	if config == nil {
		config = DefaultConfig()
	}

	return &SimClusters{
		config:         config,
		graphIndex:     NewGraphIndex(),
		MaxConcurrency: runtime.NumCPU(),
	}
}

// IngestEdges adds multiple edges to the graph
func (sc *SimClusters) IngestEdges(edges []Edge) {
	//sc.graphIndex.AddEdges(edges)
	sc.graphIndex.AddEdgesBatch(edges) // not using this for now because it uses too much extra memory
}

// ComputeClusters performs the full clustering computation
func (sc *SimClusters) ComputeClusters() error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	slog.Info("Identifying influencers...")
	influencers := sc.identifyInfluencers()

	slog.Info("Influencer count: ", "count", len(influencers))

	slog.Info("Initializing clusters...", "count", len(influencers))
	matrix := sc.initializeInfluencerEdges(influencers, 0.1)

	sc.influencerMatrix = matrix

	// Save checkpoint after expensive computation if checkpoint path is set
	if sc.CheckpointPath != "" {
		slog.Info("Saving checkpoint after initializeInfluencerEdges...")
		if err := sc.SaveCheckpoint(); err != nil {
			slog.Error("Failed to save checkpoint", "error", err)
			return fmt.Errorf("checkpoint save failed: %w", err)
		}
	}

	// influencer graph
	g := NewGraph(matrix)

	// Log graph statistics
	avgDegree := float64(2*len(g.Edges)) / float64(len(g.Nodes))
	maxPossibleEdges := len(g.Nodes) * (len(g.Nodes) - 1) / 2
	density := float64(len(g.Edges)) / float64(maxPossibleEdges)
	slog.Info("Influencer graph statistics",
		"nodes", len(g.Nodes),
		"edges", len(g.Edges),
		"avg_degree", fmt.Sprintf("%.2f", avgDegree),
		"density", fmt.Sprintf("%.6f", density))

	// Twitter SimClusters parameters (from KDD 2020 paper)
	const (
		alpha                  = 10.0 // weight for edge coverage vs non-edge separation
		maxCommPerNode         = 4    // l parameter - max communities per influencer
		epsilon                = 0.05 // random exploration probability
		maxNeighborCommunities = 4    // cap on neighborhood size for enumeration
	)

	influencerCommunities := ConcurrentMHSearchOverlapping(
		g,
		10_000,                 // numCommunities
		1_000_000,              // totalIters (1M iterations total)
		alpha,                  // alpha parameter
		maxCommPerNode,         // max communities per node
		epsilon,                // random exploration rate
		maxNeighborCommunities, // cap on neighborhood
	)

	// Count unique communities that have at least one member
	uniqueCommunities := make(map[int]bool)
	for _, comms := range influencerCommunities {
		for comm := range comms {
			uniqueCommunities[comm] = true
		}
	}

	slog.Info("Community detection complete",
		"influencers_assigned", len(influencerCommunities),
		"unique_communities", len(uniqueCommunities))

	// Save influencer communities mapping
	slog.Info("writing influencer communities...")
	commFile, err := os.Create("influencer_communities.json")
	if err != nil {
		return err
	}
	if err := json.NewEncoder(commFile).Encode(influencerCommunities); err != nil {
		commFile.Close()
		return err
	}
	commFile.Close()

	clusters := ComputeInterestedIn(sc.graphIndex, influencerCommunities, 1000, 0.1)

	slog.Info("writing output clusters...")
	fi, err := os.Create("mapping.json")
	if err != nil {
		return err
	}
	defer fi.Close()

	return json.NewEncoder(fi).Encode(clusters)
	//sc.needsFullRecompute = false
	//sc.updateEpoch++
}

// ComputeClustersFromCheckpoint loads state from checkpoint and continues with Metropolis-Hastings
func (sc *SimClusters) ComputeClustersFromCheckpoint(checkpointPath string) error {
	slog.Info("Loading checkpoint to resume computation...", "path", checkpointPath)

	state, err := LoadCheckpoint(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to load checkpoint: %w", err)
	}

	sc.mu.Lock()
	defer sc.mu.Unlock()

	// Restore state
	sc.graphIndex = state.GraphIndex
	sc.influencerMatrix = state.InfluencerMatrix
	influencers := state.Influencers

	var selfEdges int
	var sew float64
	for edge, weight := range sc.influencerMatrix {
		if edge.A == edge.B {
			selfEdges++
			sew += weight
		}
	}

	slog.Info("self edge scan results", "selfEdges", selfEdges, "sew", sew)

	slog.Info("Checkpoint loaded", "influencers", len(influencers), "matrix_edges", len(sc.influencerMatrix))

	// Continue with clustering from the checkpoint
	for k, v := range sc.influencerMatrix {
		if v < 0.4 {
			delete(sc.influencerMatrix, k)
		}
	}

	g := NewGraph(sc.influencerMatrix)

	// Twitter SimClusters parameters (from KDD 2020 paper)
	const (
		alpha                  = 10.0 // weight for edge coverage vs non-edge separation
		maxCommPerNode         = 4    // l parameter - max communities per influencer
		epsilon                = 0.05 // random exploration probability
		maxNeighborCommunities = 4    // cap on neighborhood size for enumeration
	)

	influencerCommunities := ConcurrentMHSearchOverlapping(
		g,
		10_000,                 // numCommunities
		1_000_000,              // totalIters (1M iterations total)
		alpha,                  // alpha parameter
		maxCommPerNode,         // max communities per node
		epsilon,                // random exploration rate
		maxNeighborCommunities, // cap on neighborhood
	)

	// Count unique communities that have at least one member
	uniqueCommunities := make(map[int]bool)
	for _, comms := range influencerCommunities {
		for comm := range comms {
			uniqueCommunities[comm] = true
		}
	}

	slog.Info("Community detection complete",
		"influencers_assigned", len(influencerCommunities),
		"unique_communities", len(uniqueCommunities))

	// Save influencer communities mapping
	slog.Info("writing influencer communities...")
	commFile, err := os.Create("influencer_communities.json")
	if err != nil {
		return err
	}
	if err := json.NewEncoder(commFile).Encode(influencerCommunities); err != nil {
		commFile.Close()
		return err
	}
	commFile.Close()

	clusters := ComputeInterestedIn(sc.graphIndex, influencerCommunities, 1000, 0.1)

	slog.Info("writing output clusters...")
	fi, err := os.Create("mapping.json")
	if err != nil {
		return err
	}
	defer fi.Close()

	return json.NewEncoder(fi).Encode(clusters)
}

// SaveCheckpoint saves the current state to disk
func (sc *SimClusters) SaveCheckpoint() error {
	if sc.CheckpointPath == "" {
		return fmt.Errorf("checkpoint path not set")
	}

	// Collect influencers list
	influencers := make([]int64, 0, len(sc.graphIndex.Influencers))
	for uid := range sc.graphIndex.Influencers {
		influencers = append(influencers, uid)
	}

	state := &CheckpointState{
		Influencers:      influencers,
		InfluencerMatrix: sc.influencerMatrix,
		GraphIndex:       sc.graphIndex,
	}

	return SaveCheckpoint(sc.CheckpointPath, state)
}

// identifyInfluencers finds users with follower count above threshold
func (sc *SimClusters) identifyInfluencers() []int64 {
	influencers := make([]int64, 0)

	var folcounts []int
	for _, user := range sc.graphIndex.Users {
		folcounts = append(folcounts, int(user.FollowerCount))
	}

	slices.Sort(folcounts)

	topTenPerc := folcounts[(len(folcounts)*9)/10]
	ninetyNinePerc := folcounts[(len(folcounts)*99)/100]

	slog.Info("Influencer follower threshold", "count", topTenPerc, "99perc", ninetyNinePerc)

	minFollows := 1000
	if topTenPerc < minFollows {
		topTenPerc = minFollows
	}

	for userID, user := range sc.graphIndex.Users {
		if user.FollowerCount >= int32(topTenPerc) {
			sc.graphIndex.MarkInfluencer(userID)
			influencers = append(influencers, userID)
		}
	}

	return influencers
}

// Expects normA := math.Sqrt(float64(a.GetCardinality()))
// to be passed in so we don't have to redo this operation over and over again
func BitmapCosineSimilarity(normA, normB float64, a, b *roaring64.Bitmap) float64 {
	if normA == 0 || normB == 0 {
		return 0
	}
	dotProduct := a.AndCardinality(b) // Intersection IS the dot product

	return float64(dotProduct) / float64(normA*normB)
}

type matKey struct {
	A int64
	B int64
}

func (sc *SimClusters) influencerMatrixPrecomputation(influencers []int64) (map[int64]float64, map[int64]*MinHashSignature) {
	norms := make(map[int64]float64)
	minhashes := make(map[int64]*MinHashSignature)
	var maplk sync.Mutex

	var wg sync.WaitGroup
	sema := make(chan struct{}, sc.MaxConcurrency)

	slog.Info("starting precomputation and run optimization")
	barpre := progressbar.Default(int64(len(influencers)))
	for _, uid := range influencers {
		wg.Add(1)
		sema <- struct{}{}
		go func(uidx int64) {
			defer func() {
				wg.Done()
				<-sema
			}()
			folset := sc.graphIndex.GetFollowers(uidx)
			folset.RunOptimize()
			norm := math.Sqrt(float64(folset.GetCardinality()))
			mh := ComputeMinHash(folset, 128)

			maplk.Lock()
			norms[uidx] = norm
			minhashes[uidx] = mh
			maplk.Unlock()

			barpre.Add(1)
		}(uid)
	}
	wg.Wait()
	slog.Info("precomputation and run optimization complete")

	return norms, minhashes
}

func (sc *SimClusters) initializeInfluencerEdges(influencers []int64, minThresh float64) map[matKey]float64 {
	slices.Sort(influencers)

	norms, minhashes := sc.influencerMatrixPrecomputation(influencers)

	var ilk sync.Mutex
	infMatrix := make(map[matKey]float64)

	bar := progressbar.Default(int64(len(influencers)))
	start := time.Now()
	sema := make(chan struct{}, sc.MaxConcurrency)
	var wg sync.WaitGroup

	var skipped uint64
	var underhalf, bottom90 uint64
	for i, uid := range influencers {
		wg.Add(1)
		sema <- struct{}{}
		go func(ix int, uidx int64) {
			defer wg.Done()
			defer func() {
				bar.Add(1)
				<-sema
			}()

			folset := sc.graphIndex.GetFollowers(uidx)
			normA := norms[uidx]
			mhA := minhashes[uidx]

			for _, ouid := range influencers[ix+1:] {
				mhB := minhashes[ouid]
				mhsim := EstimateSimilarity(mhA, mhB)

				if mhsim < minThresh {
					atomic.AddUint64(&skipped, 1)
					continue
				}

				ofolset := sc.graphIndex.GetFollowers(ouid)

				normB := norms[ouid]

				val := BitmapCosineSimilarity(normA, normB, folset, ofolset)

				if val < minThresh {
					continue
				}

				ilk.Lock()
				infMatrix[matKey{
					A: uidx,
					B: ouid,
				}] = val
				ilk.Unlock()
			}
		}(i, uid)
	}

	wg.Wait()
	bar.Finish()

	slog.Info("minhash skipped pairs", "count", skipped, "50perc", underhalf, "90perc", bottom90)

	var minval float64 = 100000000
	var maxval, sum float64
	for _, val := range infMatrix {
		sum += val
		if val < minval {
			minval = val
		}
		if val > maxval {
			maxval = val
		}
	}

	slog.Info("Built influencer matrix", "took", time.Since(start), "edges", len(infMatrix), "min", minval, "max", maxval, "avg", sum/float64(len(infMatrix)))
	return infMatrix
}

type neighborWeight struct {
	Node   int64
	Weight float64
}

type Graph struct {
	Nodes []int64
	Edges map[matKey]float64

	Neighbors map[int64][]neighborWeight
}

func NewGraph(edges map[matKey]float64) *Graph {
	g := &Graph{
		Edges:     edges,
		Neighbors: make(map[int64][]neighborWeight),
	}
	nodeMap := make(map[int64]bool)

	// Extract all nodes
	for key, val := range edges {
		nodeMap[key.A] = true
		nodeMap[key.B] = true

		g.Neighbors[key.A] = append(g.Neighbors[key.A], neighborWeight{
			Node:   key.B,
			Weight: val,
		})
		g.Neighbors[key.B] = append(g.Neighbors[key.B], neighborWeight{
			Node:   key.A,
			Weight: val,
		})
	}

	for nd := range nodeMap {
		g.Nodes = append(g.Nodes, nd)
	}

	for _, v := range g.Neighbors {
		slices.SortFunc(v, func(a, b neighborWeight) int {
			if a.Weight > b.Weight {
				return -1
			}
			if a.Weight < b.Weight {
				return 1
			}
			return 0
		})
	}

	return g
}

func initializeRandomAssignment(graph *Graph, numCommunities int) map[int64]int {
	assignment := make(map[int64]int)
	for _, node := range graph.Nodes {
		assignment[node] = rand.Intn(numCommunities)
	}
	return assignment
}

func copyAssignment(a map[int64]int) map[int64]int {
	nmap := make(map[int64]int, len(a))
	for k, v := range a {
		nmap[k] = v
	}
	return nmap
}

// Deprecated: Use ConcurrentMHSearchOverlapping instead
// This old version uses modularity objective and single-community assignment
func ConcurrentMHSearch_DEPRECATED(graph *Graph, numCommunities, itersPerLoop, totalLoops, concurrency int, useLabelProp bool) map[int64]int {
	// Log graph statistics
	numNodes := len(graph.Nodes)
	numEdges := len(graph.Edges)
	var totalDegree, totalEdgeWeight, minWeight, maxWeight float64 = 0, 0, math.MaxFloat64, 0
	for _, neighbors := range graph.Neighbors {
		for _, n := range neighbors {
			totalDegree += n.Weight
		}
	}
	for _, weight := range graph.Edges {
		totalEdgeWeight += weight
		if weight < minWeight {
			minWeight = weight
		}
		if weight > maxWeight {
			maxWeight = weight
		}
	}
	avgDegree := totalDegree / float64(numNodes)
	avgEdgeWeight := totalEdgeWeight / float64(numEdges)
	density := float64(numEdges) / (float64(numNodes) * float64(numNodes-1) / 2)

	slog.Info("Graph statistics",
		"nodes", numNodes,
		"edges", numEdges,
		"total_edge_weight", totalEdgeWeight,
		"avg_edge_weight", avgEdgeWeight,
		"min_edge_weight", minWeight,
		"max_edge_weight", maxWeight,
		"avg_degree", avgDegree,
		"density", density,
		"target_communities", numCommunities)

	var startingAssignment map[int64]int
	var modularity float64

	if useLabelProp {
		slog.Info("Using label propagation for initialization...")
		startingAssignment = labelPropagationInit(graph, numCommunities, 5) // can increase this more later if we like the results
		modularity = computeModularity(graph, startingAssignment, numCommunities)
		validateModularity(modularity, "label_propagation_init")
		slog.Info("Label propagation initialization complete", "modularity", modularity)
	} else {
		// start by generating some number of random starting assignments, and then picking the one with highest modularity
		slog.Info("doing a search for a decent starting position...")
		startingAssignment = initializeRandomAssignment(graph, numCommunities)
		modularity = computeModularity(graph, startingAssignment, numCommunities)
		mod2 := Modularity2(startingAssignment, graph)
		//startingAssignment, modularity = initializationSearch(graph, numCommunities, concurrency)
		validateModularity(modularity, "initialization_search")
		slog.Info("starting assignment found", "modularity", modularity, "mod2", mod2)
	}

	degree := make(map[int64]float64)
	var totalWeight float64
	for key, weight := range graph.Edges {
		degree[key.A] += weight
		degree[key.B] += weight
		totalWeight += weight
	}

	// Initialize community degrees
	communityDegrees := make(map[int]float64)
	for node, comm := range startingAssignment {
		communityDegrees[comm] += degree[node]
	}

	mhpc := &mhPrecomputes{
		degree:           degree,
		totalWeight:      totalWeight,
		communityDegrees: communityDegrees,
	}

	cur := startingAssignment

	startTemp := float64(4)
	endTemp := float64(1)

	for i := 0; i < totalLoops; i++ {
		progress := float64(i) / float64(totalLoops-1)
		temperature := startTemp * float64(math.Pow(float64(endTemp/startTemp), float64(progress)))

		next, nmod := mhSearchRound(graph, cur, modularity, numCommunities, concurrency, itersPerLoop, temperature, mhpc)

		// Validate by recomputing full modularity
		//actualMod := computeModularity(graph, next, numCommunities)
		//drift := actualMod - nmod

		validateModularity(nmod, "mh_search_round_accumulated")
		//validateModularity(actualMod, "mh_search_round_actual")

		/*
			if math.Abs(float64(drift)) > 0.001 {
				slog.Warn("Modularity drift detected", "accumulated", nmod, "actual", actualMod, "drift", drift)
			}
		*/

		slog.Info("mh search round complete", "round", i, "modularity", nmod, "delta", nmod-modularity, "temperature", temperature)

		// Use actual recomputed modularity to avoid drift
		modularity = nmod
		cur = next

		// Recompute community degrees for the new best assignment
		for comm := range mhpc.communityDegrees {
			mhpc.communityDegrees[comm] = 0
		}
		for node, comm := range cur {
			mhpc.communityDegrees[comm] += mhpc.degree[node]
		}
	}

	return cur
}

func copyCommunityDegrees(cd map[int]float64) map[int]float64 {
	copy := make(map[int]float64, len(cd))
	for k, v := range cd {
		copy[k] = v
	}
	return copy
}

func mhSearchRound(graph *Graph, assignment map[int64]int, currentScore float64, numCommunities, concurrency, numIters int, temperature float64, pc *mhPrecomputes) (map[int64]int, float64) {
	var lk sync.Mutex
	var best map[int64]int = assignment
	var bestScore float64 = currentScore

	var wg sync.WaitGroup

	for i := 0; i < concurrency; i++ {
		mcopy := copyAssignment(assignment)
		// Each worker needs its own copy of communityDegrees since they modify it
		pcCopy := &mhPrecomputes{
			degree:           pc.degree, // read-only, can share
			totalWeight:      pc.totalWeight,
			communityDegrees: copyCommunityDegrees(pc.communityDegrees),
		}

		wg.Go(func() {
			nassign, nmod := MetropolisHastings(graph, mcopy, currentScore, numCommunities, numIters, temperature, pcCopy)

			lk.Lock()
			if nmod > bestScore {
				bestScore = nmod
				best = nassign
			}
			lk.Unlock()
		})
	}

	wg.Wait()

	return best, bestScore
}

func initializationSearch(graph *Graph, numCommunities, concurrency int) (map[int64]int, float64) {
	var lk sync.Mutex
	var best map[int64]int
	var bestScore float64

	var wg sync.WaitGroup
	sema := make(chan struct{}, concurrency)

	for i := 0; i < 1000; i++ {
		sema <- struct{}{}
		wg.Go(func() {
			defer func() {
				<-sema
			}()

			assignment := initializeRandomAssignment(graph, numCommunities)
			mod := computeModularity(graph, assignment, numCommunities)

			lk.Lock()
			if mod > bestScore {
				bestScore = mod
				best = assignment
			}
			lk.Unlock()
		})
	}

	wg.Wait()

	return best, bestScore
}

type mhPrecomputes struct {
	degree           map[int64]float64
	totalWeight      float64
	communityDegrees map[int]float64 // K_c for each community
}

// Deprecated: Old single-community MetropolisHastings (for backward compatibility)
func MetropolisHastings(graph *Graph, startingAssignment map[int64]int, currentScore float64, numCommunities, maxIters int, temperature float64, pc *mhPrecomputes) (map[int64]int, float64) {
	// This is a stub for backward compatibility with deprecated code
	// New code should use MetropolisHastingsOverlapping
	return startingAssignment, currentScore
}

// MetropolisHastingsOverlapping implements Twitter's Neighborhood-aware MH
// for overlapping community detection using the F(Z) objective.
//
// Parameters:
//   - alpha: weight for edge coverage vs non-edge separation (typically 10)
//   - maxCommunitiesPerNode: max communities per node (l parameter, typically 4)
//   - epsilon: probability of random exploration (typically 0.05)
//   - maxNeighborCommunities: cap on neighborhood communities to enumerate (typically 15)
//
// Returns the final Z matrix
func MetropolisHastingsOverlapping(
	graph *Graph,
	startingZ ZMatrix,
	numCommunities int,
	maxIters int,
	alpha float64,
	maxCommunitiesPerNode int,
	epsilon float64,
	maxNeighborCommunities int,
) ZMatrix {
	Z := startingZ

	bar := progressbar.Default(int64(maxIters))

	for iter := 0; iter < maxIters; iter++ {
		bar.Add(1)
		// Pick a random node
		node := graph.Nodes[rand.Intn(len(graph.Nodes))]
		oldSet := Z[node]

		// Step 1: Collect communities from neighborhood
		neighbors := graph.Neighbors[node]
		nsets := make([]*roaring64.Bitmap, 0, len(neighbors))
		for _, neighbor := range neighbors {
			nsets = append(nsets, Z[neighbor.Node])
		}

		neighborCommunities := roaring64.ParOr(2, nsets...)

		// Step 2: Add random exploration with probability epsilon
		if rand.Float64() < epsilon {
			// Add 1-2 random communities
			numRandom := 1 + rand.Intn(2)
			for i := 0; i < numRandom && int(neighborCommunities.GetCardinality()) < maxNeighborCommunities; i++ {
				randomComm := rand.Intn(numCommunities)
				neighborCommunities.Add(uint64(randomComm))
			}
		}

		// Step 3: Cap neighborhood communities if too many
		if int(neighborCommunities.GetCardinality()) > maxNeighborCommunities {
			// Keep only maxNeighborCommunities random ones
			commList := make([]uint64, 0, neighborCommunities.GetCardinality())
			for val := range roaring64.Values(neighborCommunities) {
				commList = append(commList, val)
			}
			rand.Shuffle(len(commList), func(i, j int) {
				commList[i], commList[j] = commList[j], commList[i]
			})
			neighborCommunities.Clear()
			for i := 0; i < maxNeighborCommunities; i++ {
				neighborCommunities.Add(commList[i])
			}
		}

		if neighborCommunities.IsEmpty() {
			continue // isolated node, skip
		}

		// Step 4: Enumerate all subsets up to size maxCommunitiesPerNode
		commList := make([]uint64, 0, neighborCommunities.GetCardinality())
		iter := neighborCommunities.Iterator()
		for iter.HasNext() {
			commList = append(commList, iter.Next())
		}

		candidateSets := make([]*roaring64.Bitmap, 0, 50)
		enumerateSubsets(commList, maxCommunitiesPerNode, func(bm *roaring64.Bitmap) {
			candidateSets = append(candidateSets, bm.Clone())
		})

		// Step 5: Compute f(u, s) for each candidate subset
		scores := make([]float64, len(candidateSets))
		for i, candidateSet := range candidateSets {
			scores[i] = computeFNode(graph, Z, node, candidateSet, alpha)
		}

		// Step 6: Sample a subset via softmax (using Gumbel-Max trick)
		selectedIdx := sampleSoftmax(scores)
		newSet := candidateSets[selectedIdx]

		// Step 7: Metropolis-Hastings acceptance
		// Compute change in objective
		oldScore := computeFNode(graph, Z, node, oldSet, alpha)
		newScore := scores[selectedIdx]
		delta := newScore - oldScore

		// Accept with probability min(1, e^delta)
		acceptProb := math.Min(1.0, math.Exp(delta))
		if rand.Float64() < acceptProb {
			// Accept: update Z[node]
			Z[node] = newSet
		}
		// If rejected, Z[node] stays the same
	}

	return Z
}

// ============================================================================
// Twitter SimClusters F(Z) Objective Functions
// ============================================================================

// ZMatrix represents community assignments: node ID -> set of communities (roaring bitmap)
type ZMatrix map[int64]*roaring64.Bitmap

// hasIntersection returns true if two community sets share at least one community
func hasIntersection(a, b *roaring64.Bitmap) bool {
	return a.Intersects(b)
}

// computeF calculates the global F(Z) objective function:
// F(Z) = α·Σ_(u,v)∈E 𝟙(|Z(u)∩Z(v)| > 0) + Σ_(u,v)∉E 𝟙(|Z(u)∩Z(v)| = 0)
//
// This counts:
// - How many edge pairs share ≥1 community (weighted by alpha)
// - How many non-edge pairs share 0 communities
func computeF(graph *Graph, Z ZMatrix, alpha float64) float64 {
	var edgeTerm float64

	// Edge term: iterate over all edges
	for edge := range graph.Edges {
		zA := Z[edge.A]
		zB := Z[edge.B]
		if hasIntersection(zA, zB) {
			edgeTerm += 1.0
		}
	}

	// Non-edge term optimization:
	// Non-edges with 0 shared communities = Total non-edges - Non-edges with ≥1 shared community
	numNodes := len(graph.Nodes)
	totalPairs := numNodes * (numNodes - 1) / 2
	totalNonEdges := float64(totalPairs - len(graph.Edges))

	// Build community membership lists
	commMembers := make(map[int][]int64)
	for node, communities := range Z {
		iter := communities.Iterator()
		for iter.HasNext() {
			comm := int(iter.Next())
			commMembers[comm] = append(commMembers[comm], node)
		}
	}

	// Count pairs that share ≥1 community
	pairsSharingComm := make(map[matKey]bool)

	for _, members := range commMembers {
		// For each pair in this community
		for i := 0; i < len(members); i++ {
			for j := i + 1; j < len(members); j++ {
				a, b := members[i], members[j]
				if a > b {
					a, b = b, a
				}
				pairsSharingComm[matKey{A: a, B: b}] = true
			}
		}
	}

	// Count how many pairs sharing communities are NON-edges
	nonEdgesSharingComm := 0.0
	for pair := range pairsSharingComm {
		if _, ok := graph.Edges[pair]; ok {
			nonEdgesSharingComm += 1.0
		}
	}

	// Non-edges that DON'T share any community
	nonEdgeTerm := totalNonEdges - nonEdgesSharingComm

	return alpha*edgeTerm + nonEdgeTerm
}

// computeFNode calculates the local objective f(u, s) for a single node:
// f(u, s) = α·Σ_{v∈N(u)} 𝟙(|s∩Z(v)| > 0) + Σ_{v∉N(u)} 𝟙(|s∩Z(v)| = 0)
//
// This evaluates how good it is to assign node u to community set s
// given the current assignments Z of all other nodes.
func computeFNode(graph *Graph, Z ZMatrix, node int64, s *roaring64.Bitmap, alpha float64) float64 {
	if s.IsEmpty() {
		// Empty set: no shared communities with anyone
		// Edge term = 0 (no overlap with neighbors)
		// Non-edge term = count of all non-neighbors
		return float64(len(graph.Nodes) - len(graph.Neighbors[node]) - 1)
	}

	var edgeTerm, nonEdgeTerm float64

	// Track which nodes we've seen as neighbors
	neighborSet := make(map[int64]bool)

	// Edge term: check neighbors
	for _, neighbor := range graph.Neighbors[node] {
		neighborSet[neighbor.Node] = true
		if neighbor.Node == node {
			continue // skip self-loops
		}
		zNeighbor := Z[neighbor.Node]
		if hasIntersection(s, zNeighbor) {
			edgeTerm += 1.0
		}
	}

	// Non-edge term: check non-neighbors
	for _, otherNode := range graph.Nodes {
		if otherNode == node {
			continue
		}
		if neighborSet[otherNode] {
			continue // this is a neighbor, skip
		}
		zOther := Z[otherNode]
		if !hasIntersection(s, zOther) {
			nonEdgeTerm += 1.0
		}
	}

	return alpha*edgeTerm + nonEdgeTerm
}

// enumerateSubsets generates all subsets of communities with size <= maxSize
// Returns list of community sets as roaring bitmaps
func enumerateSubsets(communities []uint64, maxSize int, cb func(bm *roaring64.Bitmap)) {
	if maxSize > len(communities) {
		maxSize = len(communities)
	}

	bm := roaring64.New()

	var permute func(n int, offs int)
	permute = func(n int, offs int) {
		cb(bm)
		if n == 0 {
			return
		}
		for x := offs; x < len(communities); x++ {
			bm.Add(communities[x])
			permute(n-1, offs+1)
			bm.Remove(communities[x])
		}
	}

	permute(maxSize, 0)
}

// sampleSoftmax samples an index from a list of scores using softmax distribution
// Uses Gumbel-Max trick for efficient one-pass sampling
func sampleSoftmax(scores []float64) int {
	if len(scores) == 0 {
		return -1
	}
	if len(scores) == 1 {
		return 0
	}

	// Gumbel-Max trick: argmax_i (score_i + Gumbel(0,1))
	// where Gumbel(0,1) = -log(-log(Uniform(0,1)))
	maxIdx := 0
	maxVal := scores[0] - math.Log(-math.Log(rand.Float64()))

	for i := 1; i < len(scores); i++ {
		gumbel := -math.Log(-math.Log(rand.Float64()))
		val := scores[i] + gumbel
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return maxIdx
}

// initializeZNeighborhood initializes Z matrix using neighborhood seeding
// Each community is initialized as the neighborhood of a randomly selected node
// This is Algorithm 3 lines 2-4 from the SimClusters paper
func initializeZNeighborhood(graph *Graph, numCommunities int) ZMatrix {
	Z := make(ZMatrix)

	// Initialize all nodes with empty bitmaps
	for _, node := range graph.Nodes {
		Z[node] = roaring64.New()
	}

	// For each community, pick a random node and assign its neighborhood to that community
	for commID := 0; commID < numCommunities; commID++ {
		// Pick random seed node
		seedNode := graph.Nodes[rand.Intn(len(graph.Nodes))]

		// Assign seed node to this community
		Z[seedNode].Add(uint64(commID))

		// Assign all neighbors of seed to this community
		for _, neighbor := range graph.Neighbors[seedNode] {
			if neighbor.Node != seedNode {
				Z[neighbor.Node].Add(uint64(commID))
			}
		}
	}

	return Z
}

// ConcurrentMHSearchOverlapping runs the complete overlapping community detection pipeline
// using Twitter's F(Z) objective function.
//
// Returns: map[node_id] -> map[community_id] -> score
func ConcurrentMHSearchOverlapping(
	graph *Graph,
	numCommunities int,
	totalIters int,
	alpha float64,
	maxCommunitiesPerNode int,
	epsilon float64,
	maxNeighborCommunities int,
) map[int64]map[int]float64 {
	slog.Info("Starting overlapping community detection with F(Z) objective",
		"nodes", len(graph.Nodes),
		"edges", len(graph.Edges),
		"communities", numCommunities,
		"alpha", alpha,
		"maxCommPerNode", maxCommunitiesPerNode,
		"epsilon", epsilon)

	// Initialize Z using neighborhood seeding
	Z := initializeZNeighborhood(graph, numCommunities)

	// Compute initial F(Z)
	initialScore := computeF(graph, Z, alpha)
	slog.Info("Initialization complete", "F(Z)", initialScore)

	// Run Metropolis-Hastings
	finalZ := MetropolisHastingsOverlapping(
		graph,
		Z,
		numCommunities,
		totalIters,
		alpha,
		maxCommunitiesPerNode,
		epsilon,
		maxNeighborCommunities,
	)

	// Compute final F(Z)
	finalScore := computeF(graph, finalZ, alpha)
	slog.Info("MH optimization complete", "final_F(Z)", finalScore, "improvement", finalScore-initialScore)

	// Convert binary Z to weighted scores
	// For now, we'll use a simple normalization: score = 1.0 / |Z(u)| for each community
	// This gives equal weight to all communities a node belongs to
	result := make(map[int64]map[int]float64)
	for node, communities := range finalZ {
		if communities.IsEmpty() {
			continue // skip nodes with no community assignment
		}
		result[node] = make(map[int]float64)
		cardinality := communities.GetCardinality()
		weight := 1.0 / float64(cardinality)
		iter := communities.Iterator()
		for iter.HasNext() {
			comm := int(iter.Next())
			result[node][comm] = weight
		}
	}

	// Log statistics
	totalAssignments := 0
	for _, comms := range result {
		totalAssignments += len(comms)
	}
	avgCommPerNode := float64(totalAssignments) / float64(len(result))
	slog.Info("Community assignment statistics",
		"nodes_with_communities", len(result),
		"total_assignments", totalAssignments,
		"avg_communities_per_node", avgCommPerNode)

	return result
}

func computeModularity(graph *Graph, assignment map[int64]int, numCommunities int) float64 {
	// Use community-aggregated formula: Q = Σ_c [ L_c/(2m) - (K_c/(2m))² ]
	// This is O(edges + nodes) instead of O(nodes²)

	// Compute node degrees
	// Use float64 for accumulation to avoid precision loss with large graphs
	var totalWeight float64
	degree := make(map[int64]float64)

	// Populate degree map and calculate total weight
	for key, weight := range graph.Edges {
		totalWeight += weight
		degree[key.A] += weight
		degree[key.B] += weight
	}

	if totalWeight == 0 {
		return 0
	}

	// Compute L_c (sum of edge weights within community c)
	edgesWithin := make(map[int]float64)
	for key, weight := range graph.Edges {
		if assignment[key.A] == assignment[key.B] {
			edgesWithin[assignment[key.A]] += weight
		}
	}

	// Compute K_c (sum of degrees in community c)
	totalDegree := make(map[int]float64)
	for node, comm := range assignment {
		totalDegree[comm] += degree[node]
	}

	// Compute modularity: Q = Σ_c [ L_c/(2m) - (K_c/(2m))² ]
	// Note: m is the sum of edge weights (each edge counted once)
	// K_c is the sum of degrees, so sum of all K_c = 2m
	var modularity float64
	twoM := 2 * totalWeight
	for comm := range totalDegree {
		Lc := edgesWithin[comm]
		Kc := totalDegree[comm]
		modularity += float64(float64(Lc)/twoM - (float64(Kc)/twoM)*(float64(Kc)/twoM))
	}

	return modularity
}

func Modularity2(communities map[int64]int, graph *Graph) float64 {
	if len(graph.Edges) == 0 {
		return 0.0
	}

	// Calculate total weight of all edges (2m in the formula)
	var totalWeight float64
	for _, weight := range graph.Edges {
		totalWeight += weight
	}

	// Calculate degree (strength) of each node
	degrees := make(map[int64]float64)
	for edge, weight := range graph.Edges {
		degrees[edge.A] += weight
		degrees[edge.B] += weight
	}

	// Calculate modularity Q
	var q float64
	for edge, weight := range graph.Edges {
		// Check if both nodes are in the same community
		commA, okA := communities[edge.A]
		commB, okB := communities[edge.B]

		if !okA || !okB {
			// Skip edges where nodes don't have community assignments
			continue
		}

		if commA != commB {
			continue
		}

		// Q = 1/(2m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)
		// where A_ij is the weight of edge (i,j)
		// k_i and k_j are the sum of weights of edges attached to nodes i and j
		// δ(c_i, c_j) = 1 if i and j are in the same community, 0 otherwise
		expectedWeight := (degrees[edge.A] * degrees[edge.B]) / totalWeight

		q += (weight - expectedWeight)
	}

	// Normalize by total weight
	q /= totalWeight

	return q
}

func validateModularity(modularity float64, context string) {
	if modularity < -1.0 || modularity > 1.0 {
		slog.Warn("Modularity out of expected range [-1, 1]",
			"context", context,
			"modularity", modularity)
	}
	if modularity < -0.5 {
		slog.Warn("Very low modularity suggests poor community structure",
			"context", context,
			"modularity", modularity)
	}
}

func computeModularityDelta(
	graph *Graph,
	assignment map[int64]int,
	node int64,
	oldCommunity, newCommunity int,
	pc *mhPrecomputes,
) float64 {
	if pc.totalWeight == 0 {
		return 0
	}

	// Delta formula for moving node v from c_old to c_new:
	// ΔQ = (dLnew - dLold)/(2m) + 2*k*(Kold - Knew - k)/(2m)²
	// where:
	//   dLold = edges from v to nodes in old community
	//   dLnew = edges from v to nodes in new community
	//   k = degree of node v
	//   Kold = total degree of old community (includes k)
	//   Knew = total degree of new community (doesn't include k)
	//   m = sum of all edge weights

	k := pc.degree[node]
	Kold := pc.communityDegrees[oldCommunity]
	Knew := pc.communityDegrees[newCommunity]

	// Compute dLold and dLnew (edge weight changes)
	var dLold, dLnew float64
	for _, neighbor := range graph.Neighbors[node] {
		neighborComm := assignment[neighbor.Node]
		if neighborComm == oldCommunity { // && neighbor.Node != node { // this extra condition should be impossible
			dLold += neighbor.Weight
		}
		if neighborComm == newCommunity {
			dLnew += neighbor.Weight
		}
	}

	// Apply corrected formula with factor of 2 on K² term
	twoM := 2 * pc.totalWeight
	delta := (dLnew-dLold)/twoM + 2*k*(Kold-Knew-k)/(twoM*twoM)

	return delta
}

func ComputeInterestedIn(gi *GraphIndex, knownFor map[int64]map[int]float64, numCommunities int, minThresh float64) map[int64]map[int]float64 {
	var lk sync.Mutex
	interestedIn := make(map[int64]map[int]float64)

	var wg sync.WaitGroup

	sema := make(chan struct{}, 100)

	// For each consumer (user who follows others)
	gi.Following.Range(func(consumer int64, producers *roaring64.Bitmap) bool {
		wg.Add(1)
		sema <- struct{}{}
		go func(cons int64, folmap *roaring64.Bitmap) {
			defer func() {
				<-sema
				wg.Done()
			}()

			interests := make(map[int]float64)
			total := 0.0
			// For each producer this consumer follows
			iter := folmap.Iterator()
			for iter.HasNext() {
				fol := int64(iter.Next())
				if communities, exists := knownFor[fol]; exists {
					// Influencer may be in multiple communities with different scores
					for comm, score := range communities {
						interests[comm] += score
						total += score
					}
				}
			}

			if total > 0 {
				for comm, v := range interests {
					v /= total
					if v < minThresh {
						delete(interests, comm)
					} else {
						interests[comm] = v
					}
				}
			}

			if len(interests) > 0 {
				lk.Lock()
				interestedIn[cons] = interests
				lk.Unlock()
			}
		}(consumer, producers)
		return true
	})

	wg.Wait()

	return interestedIn
}

// GetNumUsers returns the number of users
func (sc *SimClusters) GetNumUsers() int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	return len(sc.graphIndex.Users)
}

// GetNumInfluencers returns the number of influencers
func (sc *SimClusters) GetNumInfluencers() int {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	return len(sc.graphIndex.Influencers)
}

// labelPropagationInit initializes communities using label propagation
// Each node starts with its own label, then iteratively adopts the most common label among neighbors
func labelPropagationInit(graph *Graph, numCommunities int, maxIters int) map[int64]int {
	slog.Info("Starting label propagation", "nodes", len(graph.Nodes), "maxIters", maxIters)

	// Initialize: each node gets its own label (its ID)
	labels := make(map[int64]int64)
	for _, node := range graph.Nodes {
		labels[node] = node
	}

	// Run label propagation iterations
	for iter := 0; iter < maxIters; iter++ {
		changed := 0

		// Randomize order each iteration to break ties
		nodes := append([]int64{}, graph.Nodes...)
		rand.Shuffle(len(nodes), func(i, j int) {
			nodes[i], nodes[j] = nodes[j], nodes[i]
		})

		// For each node, find most common label among neighbors (weighted by edge weight)
		for _, node := range nodes {
			// Count weighted label frequencies among neighbors
			labelWeights := make(map[int64]float64)
			for _, neighbor := range graph.Neighbors[node] {
				neighborLabel := labels[neighbor.Node]
				labelWeights[neighborLabel] += neighbor.Weight
			}

			// Find most common label (with highest total weight)
			var bestLabel int64
			var bestWeight float64
			for label, weight := range labelWeights {
				if weight > bestWeight || (weight == bestWeight && label < bestLabel) {
					bestWeight = weight
					bestLabel = label
				}
			}

			// Update if changed
			if bestLabel != labels[node] && bestWeight > 0 {
				labels[node] = bestLabel
				changed++
			}
		}

		slog.Info("Label propagation iteration", "iter", iter, "changed", changed)

		// Stop if converged
		if changed == 0 {
			slog.Info("Label propagation converged", "iter", iter)
			break
		}
	}

	// Compress labels to community IDs 0..numCommunities-1
	return compressLabels(labels, numCommunities)
}

// compressLabels maps arbitrary labels to community IDs 0..numCommunities-1
// Communities are ordered by size (largest first) and truncated to numCommunities
func compressLabels(labels map[int64]int64, numCommunities int) map[int64]int {
	// Count label frequencies
	labelCounts := make(map[int64]int)
	for _, label := range labels {
		labelCounts[label]++
	}

	// Sort labels by frequency (largest first)
	type labelCount struct {
		label int64
		count int
	}
	sortedLabels := make([]labelCount, 0, len(labelCounts))
	for label, count := range labelCounts {
		sortedLabels = append(sortedLabels, labelCount{label, count})
	}
	slices.SortFunc(sortedLabels, func(a, b labelCount) int {
		if a.count != b.count {
			return b.count - a.count // Descending by count
		}
		if a.label < b.label {
			return -1
		}
		return 1
	})

	// Map top numCommunities labels to 0..numCommunities-1
	// Remaining labels get assigned to communities round-robin
	labelToComm := make(map[int64]int)
	for i, lc := range sortedLabels {
		if i < numCommunities {
			labelToComm[lc.label] = i
		} else {
			// Overflow: assign round-robin to existing communities
			labelToComm[lc.label] = i % numCommunities
		}
	}

	// Map all nodes to their community IDs
	assignment := make(map[int64]int, len(labels))
	for node, label := range labels {
		assignment[node] = labelToComm[label]
	}

	slog.Info("Label compression complete",
		"unique_labels", len(labelCounts),
		"communities", numCommunities,
		"largest_community", sortedLabels[0].count)

	return assignment
}
