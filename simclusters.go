package simclusters

import (
	"encoding/json"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	"github.com/schollz/progressbar/v3"
)

// SimClusters is the main structure for the SimClusters algorithm
type SimClusters struct {
	config *Config
	mu     sync.RWMutex

	// Indices
	graphIndex *GraphIndex

	MaxConcurrency int

	// State tracking
	needsFullRecompute bool
	updateEpoch        int64
}

// NewSimClusters creates a new SimClusters instance
func NewSimClusters(config *Config) *SimClusters {
	if config == nil {
		config = DefaultConfig()
	}

	return &SimClusters{
		config:             config,
		graphIndex:         NewGraphIndex(),
		needsFullRecompute: true,
		updateEpoch:        0,
		MaxConcurrency:     runtime.NumCPU(),
	}
}

// IngestEdges adds multiple edges to the graph
func (sc *SimClusters) IngestEdges(edges []Edge) {
	sc.graphIndex.AddEdges(edges)
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

	// influencer graph
	g := NewGraph(matrix)

	influencerCommunities := MetropolisHastings(g, 1000, 10000)

	slog.Info("computed influencer communities", "count", len(influencerCommunities))

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
func BitmapCosineSimilarity(normA float64, a, b *roaring64.Bitmap) float32 {

	//normA := math.Sqrt(float64(a.GetCardinality())) // sqrt of count
	normB := math.Sqrt(float64(b.GetCardinality()))

	if normA == 0 || normB == 0 {
		return 0
	}
	dotProduct := a.AndCardinality(b) // Intersection IS the dot product

	return float32(dotProduct) / float32(normA*normB)
}

type matKey struct {
	A int64
	B int64
}

func (sc *SimClusters) initializeInfluencerEdges(influencers []int64, minThresh float32) map[matKey]float32 {
	slices.Sort(influencers)

	var ilk sync.Mutex
	infMatrix := make(map[matKey]float32)

	bar := progressbar.Default(int64(len(influencers)))
	start := time.Now()
	sema := make(chan struct{}, sc.MaxConcurrency)
	var wg sync.WaitGroup
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
			normA := math.Sqrt(float64(folset.GetCardinality()))

			for _, ouid := range influencers[ix+1:] {
				ofolset := sc.graphIndex.GetFollowers(ouid)

				val := BitmapCosineSimilarity(normA, folset, ofolset)

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

	var minval float32 = 100000000
	var maxval, sum float32
	for _, val := range infMatrix {
		sum += val
		if val < minval {
			minval = val
		}
		if val > maxval {
			maxval = val
		}
	}

	slog.Info("Built influencer matrix", "took", time.Since(start), "edges", len(infMatrix), "min", minval, "max", maxval, "avg", sum/float32(len(infMatrix)))
	return infMatrix
}

type neighborWeight struct {
	Node   int64
	Weight float32
}

type Graph struct {
	Nodes []int64
	Edges map[matKey]float32

	Neighbors map[int64][]neighborWeight
}

func NewGraph(edges map[matKey]float32) *Graph {
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

	return g
}

func initializeRandomAssignment(graph *Graph, numCommunities int) map[int64]int {
	assignment := make(map[int64]int)
	for _, node := range graph.Nodes {
		assignment[node] = rand.Intn(numCommunities)
	}
	return assignment
}

func MetropolisHastings(graph *Graph, numCommunities, maxIters int) map[int64]int {
	temperature := float32(1.0)
	//rand.Seed(config.RandomSeed)

	// Initialize random community assignments
	assignment := initializeRandomAssignment(graph, numCommunities)

	// Precompute node degrees and total weight (these don't change)
	degree := make(map[int64]float32)
	var totalWeight float32
	for key, weight := range graph.Edges {
		degree[key.A] += weight
		degree[key.B] += weight
		totalWeight += weight
	}

	// Current score
	currentScore := computeModularity(graph, assignment, numCommunities)

	// Metropolis-Hastings sampling
	for iter := 0; iter < maxIters; iter++ {
		// Propose a move: pick a random node and random new community
		node := graph.Nodes[rand.Intn(len(graph.Nodes))]
		oldCommunity := assignment[node]
		newCommunity := rand.Intn(numCommunities)

		// Skip if same community
		if oldCommunity == newCommunity {
			continue
		}

		// Compute delta modularity (change from moving this node)
		deltaScore := computeModularityDelta(graph, assignment, node, oldCommunity, newCommunity, degree, totalWeight)

		// Acceptance probability
		acceptProb := float32(math.Min(1.0, math.Exp(float64(deltaScore/temperature))))

		// Accept or reject
		if rand.Float32() < acceptProb {
			// Accept the move
			assignment[node] = newCommunity
			currentScore += deltaScore
		}
		// If rejected, assignment stays the same (no revert needed)
	}

	return assignment
}

func computeModularity(graph *Graph, assignment map[int64]int, numCommunities int) float32 {
	// Total weight of all edges
	var totalWeight float32
	for _, weight := range graph.Edges {
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0
	}

	// Compute node degrees (sum of edge weights for each node)
	degree := make(map[int64]float32)
	for key, weight := range graph.Edges {
		degree[key.A] += weight
		degree[key.B] += weight
	}

	var modularity float32

	// For each edge, check if both endpoints are in the same community
	for key, weight := range graph.Edges {
		if assignment[key.A] == assignment[key.B] {
			// Both nodes in same community
			expected := (degree[key.A] * degree[key.B]) / (2 * totalWeight)
			modularity += weight - expected
		}
	}

	return modularity / (2 * totalWeight)
}

func computeModularityDelta(
	graph *Graph,
	assignment map[int64]int,
	node int64,
	oldCommunity, newCommunity int,
	degree map[int64]float32,
	totalWeight float32,
) float32 {
	if totalWeight == 0 {
		return 0
	}

	var delta float32
	neighbors := graph.Neighbors[node]

	// For each neighbor, check how the edge contribution changes
	for _, n := range neighbors {
		neighborCommunity := assignment[n.Node]

		// Loss from old community: if neighbor was in old community, we lose this edge
		if neighborCommunity == oldCommunity {
			expected := (degree[node] * degree[n.Node]) / (2 * totalWeight)
			delta -= (n.Weight - expected) / (2 * totalWeight)
		}

		// Gain from new community: if neighbor is in new community, we gain this edge
		if neighborCommunity == newCommunity {
			expected := (degree[node] * degree[n.Node]) / (2 * totalWeight)
			delta += (n.Weight - expected) / (2 * totalWeight)
		}
	}

	return delta
}

func ComputeInterestedIn(gi *GraphIndex, knownFor map[int64]int, numCommunities int, minThresh float64) map[int64]map[int]float64 {
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
				if community, exists := knownFor[fol]; exists {
					interests[community] += 1
					total += 1
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
