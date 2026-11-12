package simclusters

import (
	"math"
	"math/rand"
	"slices"
	"sync"
	"testing"
)

func TestAddEdgesBatchConcurrentFirstInsert(t *testing.T) {
	// Many goroutines creating the same key at once used to lose edges.
	const goroutines = 32
	for trial := 0; trial < 300; trial++ {
		gi := NewGraphIndex()
		var wg sync.WaitGroup
		start := make(chan struct{})
		for g := 0; g < goroutines; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				<-start
				gi.AddEdgesBatch([]Edge{{From: int64(100 + g), To: 1}, {From: 7, To: int64(100 + g)}})
			}(g)
		}
		close(start)
		wg.Wait()
		if got := gi.FollowerCount(1); got != goroutines {
			t.Fatalf("trial %d: user 1 has %d followers, want %d", trial, got, goroutines)
		}
		if got := len(gi.Following(7)); got != goroutines {
			t.Fatalf("trial %d: user 7 follows %d users, want %d", trial, got, goroutines)
		}
	}
}

func TestFollowerCountIgnoresDuplicateEdges(t *testing.T) {
	gi := NewGraphIndex()
	gi.AddEdgesBatch([]Edge{{From: 5, To: 1}, {From: 5, To: 1}, {From: 6, To: 1}})
	gi.AddEdgesBatch([]Edge{{From: 5, To: 1}})
	if got := gi.FollowerCount(1); got != 2 {
		t.Errorf("FollowerCount(1) = %d, want 2", got)
	}
	if got := gi.FollowerCount(99); got != 0 {
		t.Errorf("FollowerCount of an unknown user = %d, want 0", got)
	}
}

// followEdges gives influencer `to` the followers base..base+n-1.
func followEdges(edges []Edge, to int64, base, n int) []Edge {
	for i := 0; i < n; i++ {
		edges = append(edges, Edge{From: int64(base + i), To: to})
	}
	return edges
}

func TestSimilarityKeepsPairsMinHashUsedToDrop(t *testing.T) {
	// The old MinHash prefilter compared a Jaccard estimate against the
	// cosine threshold, which dropped most of these pairs. The last one sits
	// exactly on the threshold.
	for _, tc := range []struct{ a, b, inter int }{
		{10_000, 100_000, 9_000}, // cosine 0.285, Jaccard 0.089
		{10_000, 10_000, 1_500},  // cosine 0.150, Jaccard 0.081
		{2_000, 150_000, 2_000},  // cosine 0.115, Jaccard 0.013
		{1, 100, 1},              // cosine 0.100
	} {
		const userBase = 1_000_000
		gi := NewGraphIndex()
		edges := followEdges(nil, 1, userBase, tc.a)
		edges = followEdges(edges, 2, userBase+tc.a-tc.inter, tc.b)
		gi.AddEdgesBatch(edges)

		want := float64(tc.inter) / math.Sqrt(float64(tc.a)*float64(tc.b))
		got := slices.Collect(buildSimilarity(gi, []int64{2, 1}, 0.1, 2, nil).edges())
		if len(got) != 1 || got[0].A != 1 || got[0].B != 2 || math.Abs(got[0].Weight-want) > 1e-12 {
			t.Errorf("sizes %d/%d: got %+v, want one edge 1-2 with weight %.4f", tc.a, tc.b, got, want)
		}
	}
}

func TestSimilarityEdgesMatchBruteForce(t *testing.T) {
	// Influencers draw followers from a shared pool per topic, which yields
	// many pairs on both sides of the threshold and very unequal set sizes.
	const (
		threshold   = 0.1
		topics      = 6
		perTopic    = 25
		poolSize    = 3000
		influencers = topics * perTopic
	)
	r := rand.New(rand.NewSource(2))
	gi := NewGraphIndex()
	ids := make([]int64, influencers)
	var edges []Edge
	for i := range ids {
		ids[i] = int64(i + 1)
		pool := 1_000_000 * (1 + i%topics)
		size := 100 + r.Intn(2500)
		for _, u := range r.Perm(poolSize)[:size] {
			edges = append(edges, Edge{From: int64(pool + u), To: ids[i]})
		}
		for j := 0; j < 50; j++ { // a few followers from anywhere
			edges = append(edges, Edge{From: int64(r.Intn(8_000_000)), To: ids[i]})
		}
		if i%10 == 0 { // influencers follow each other too
			edges = append(edges, Edge{From: ids[i], To: ids[(i+1)%influencers]})
		}
	}
	gi.AddEdgesBatch(edges)

	followers := make(map[int64]map[int64]bool)
	for _, e := range edges {
		if followers[e.To] == nil {
			followers[e.To] = make(map[int64]bool)
		}
		followers[e.To][e.From] = true
	}
	var want []SimEdge
	for i, a := range ids {
		for _, b := range ids[i+1:] {
			fa, fb := followers[a], followers[b]
			inter := 0
			for u := range fa {
				if fb[u] {
					inter++
				}
			}
			sim := cosineSimilarity(float64(inter), float64(len(fa)), float64(len(fb)))
			if sim >= threshold {
				want = append(want, SimEdge{A: a, B: b, Weight: sim})
			}
		}
	}
	if len(want) < 1000 {
		t.Fatalf("test graph has only %d similar pairs", len(want))
	}

	for _, workers := range []int{1, 5} {
		got := slices.Collect(buildSimilarity(gi, ids, threshold, workers, nil).edges())
		if !slices.Equal(got, want) {
			t.Errorf("workers=%d: got %d edges, brute force found %d (or they differ)", workers, len(got), len(want))
		}
	}
}
