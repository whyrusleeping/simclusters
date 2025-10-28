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
		if v < 0.2 {
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

	minFollows := 1500
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
			mh := ComputeMinHash(folset, 200)

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

			lmatrix := make(map[matKey]float64, len(influencers)-ix)
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

				lmatrix[matKey{
					A: uidx,
					B: ouid,
				}] = val
			}

			ilk.Lock()
			for k, v := range lmatrix {
				infMatrix[k] = v
			}
			ilk.Unlock()
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

	Neighbors    map[int64][]neighborWeight
	NeighborSets map[int64]map[int64]float64
}

func NewGraph(edges map[matKey]float64) *Graph {
	g := &Graph{
		Edges:        edges,
		Neighbors:    make(map[int64][]neighborWeight),
		NeighborSets: make(map[int64]map[int64]float64),
	}
	nodeMap := make(map[int64]bool)

	// Extract all nodes
	var wg sync.WaitGroup

	wg.Go(func() {
		for key, _ := range edges {
			nodeMap[key.A] = true
			nodeMap[key.B] = true
		}
	})

	wg.Go(func() {
		for key, val := range edges {
			g.Neighbors[key.A] = append(g.Neighbors[key.A], neighborWeight{
				Node:   key.B,
				Weight: val,
			})
			g.Neighbors[key.B] = append(g.Neighbors[key.B], neighborWeight{
				Node:   key.A,
				Weight: val,
			})
		}
	})

	wg.Go(func() {
		for key, val := range edges {
			nsa, ok := g.NeighborSets[key.A]
			if !ok {
				nsa = make(map[int64]float64)
				g.NeighborSets[key.A] = nsa
			}
			nsa[key.B] = val

			nsb, ok := g.NeighborSets[key.B]
			if !ok {
				nsb = make(map[int64]float64)
				g.NeighborSets[key.B] = nsb
			}
			nsb[key.A] = val
		}
	})

	wg.Wait()

	for nd := range nodeMap {
		g.Nodes = append(g.Nodes, nd)
	}

	/*
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
	*/

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

func copyCommunityDegrees(cd map[int]float64) map[int]float64 {
	copy := make(map[int]float64, len(cd))
	for k, v := range cd {
		copy[k] = v
	}
	return copy
}

type communityStat struct {
	neighborWeight   float64
	nonNeighborCount float64
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

	// map from communities to users in them
	Zinvert := make(map[uint64]map[int64]bool)

	for u, cset := range Z {
		iter := cset.Iterator()
		for iter.HasNext() {
			c := iter.Next()
			umap, ok := Zinvert[c]
			if !ok {
				umap = make(map[int64]bool)
				Zinvert[c] = umap
			}
			umap[u] = true
		}
	}

	bar := progressbar.Default(int64(maxIters))

	var deltaGain float64

	var totalCsets int64
	var averageCsetSize int

	for iter := 0; iter < maxIters; iter++ {
		bar.Add(1)
		bar.Describe(fmt.Sprintf("dg: %0.1f, avcss: %d", deltaGain, averageCsetSize))
		// Pick a random node
		node := graph.Nodes[rand.Intn(len(graph.Nodes))]
		oldSet := Z[node]

		// Step 1: Collect communities from neighborhood
		neighbors := graph.Neighbors[node]
		nsets := make([]*roaring64.Bitmap, 0, len(neighbors))
		for _, neighbor := range neighbors {
			comms := Z[neighbor.Node]
			nsets = append(nsets, comms)
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
		neighborCommunities = limitNeighborhood(neighborCommunities, maxNeighborCommunities)

		if neighborCommunities.IsEmpty() {
			continue // isolated node, skip
		}

		// Step 4: Enumerate all subsets up to size maxCommunitiesPerNode
		commList := neighborCommunities.ToArray()

		candidateSets := make([]*roaring64.Bitmap, 0, 16)
		enumerateSubsets(commList, maxCommunitiesPerNode, func(bm *roaring64.Bitmap) {
			candidateSets = append(candidateSets, bm.Clone())
		})

		totalCsets += int64(len(candidateSets))
		averageCsetSize = int(totalCsets) / int(iter+1)

		// Step 5: Precompute per-community statistics ONCE, reuse for all candidates
		// This avoids iterating through Zinv[comm] users multiple times
		neighborSet := graph.NeighborSets[node]
		nonEdgeScore := 1.0 / alpha

		// Cache: for each community, store (neighbor_weight_sum, non_neighbor_count)
		commStats := make(map[uint64]communityStat)

		// Build union of all communities we'll need (from oldSet and all candidates)
		allRelevantComms := oldSet.Clone()
		for _, cset := range candidateSets {
			allRelevantComms.Or(cset)
		}

		// Compute statistics once per community

		var rcwg sync.WaitGroup
		var cslk sync.Mutex

		rciter := allRelevantComms.Iterator()
		for rciter.HasNext() {
			comm := rciter.Next()
			rcwg.Go(func() {
				users, ok := Zinvert[comm]
				if !ok {
					return
				}

				var neighborWeight float64
				var nonNeighborCount float64

				for user := range users {
					if nw, isNeighbor := neighborSet[user]; isNeighbor {
						neighborWeight += nw
					} else {
						nonNeighborCount += 1.0
					}
				}

				cslk.Lock()
				commStats[comm] = struct {
					neighborWeight   float64
					nonNeighborCount float64
				}{neighborWeight, nonNeighborCount}
				cslk.Unlock()
			})
		}
		rcwg.Wait()

		// Step 6: Compute deltas using cached statistics
		scores := make([]float64, len(candidateSets))
		for i, candidateSet := range candidateSets {
			var delta float64

			// Communities we're leaving
			oiter := oldSet.Iterator()
			for oiter.HasNext() {
				oc := oiter.Next()
				if !candidateSet.Contains(oc) {
					stats := commStats[oc]
					delta -= stats.neighborWeight
					delta += stats.nonNeighborCount * nonEdgeScore
				}
			}

			// Communities we're joining
			niter := candidateSet.Iterator()
			for niter.HasNext() {
				nc := niter.Next()
				if !oldSet.Contains(nc) {
					stats := commStats[nc]
					delta += stats.neighborWeight
					delta -= stats.nonNeighborCount * nonEdgeScore
				}
			}

			scores[i] = delta
		}

		// Step 7: Sample a subset via softmax (using Gumbel-Max trick)
		selectedIdx := sampleSoftmax(scores)
		newSet := candidateSets[selectedIdx]

		// Step 8: Metropolis-Hastings acceptance
		// Compute change in objective
		delta := scores[selectedIdx]

		// Accept with probability min(1, e^delta)
		acceptProb := math.Min(1.0, math.Exp(delta))
		if rand.Float64() < acceptProb {
			// Accept: update Z[node]
			Z[node] = newSet
			updateZinv(Zinvert, node, oldSet, newSet)
			deltaGain += delta
		}
		// If rejected, Z[node] stays the same
	}

	return Z
}

func limitNeighborhood(neighborCommunities *roaring64.Bitmap, maxNeighborCommunities int) *roaring64.Bitmap {
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
	return neighborCommunities
}

func updateZinv(Zinvert map[uint64]map[int64]bool, node int64, oldSet, newSet *roaring64.Bitmap) {
	// Update Zinvert: remove node from old communities, add to new ones
	// Remove from old communities
	iter := oldSet.Iterator()
	for iter.HasNext() {
		oldComm := iter.Next()
		if !newSet.Contains(oldComm) {
			// This community is being left
			if users, ok := Zinvert[oldComm]; ok {
				delete(users, node)
			}
		}
	}

	// Add to new communities
	newIter := newSet.Iterator()
	for newIter.HasNext() {
		newComm := newIter.Next()
		if !oldSet.Contains(newComm) {
			// This community is being joined
			if _, ok := Zinvert[newComm]; !ok {
				Zinvert[newComm] = make(map[int64]bool)
			}
			Zinvert[newComm][node] = true
		}
	}
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
			permute(n-1, x+1)
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
	//initialScore := computeF(graph, Z, alpha)
	//slog.Info("Initialization complete", "F(Z)", initialScore)
	initialScore := -1.0

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
	//finalScore := computeF(graph, finalZ, alpha)
	finalScore := -1.0
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
