package simclusters

import (
	"maps"
	"math"
	"math/rand"
	"slices"
	"testing"
)

// plantedEdges builds a similarity graph of groups nodes-per-group cliques
// (each intra-group pair is an edge with probability pIn) plus noise edges
// between random nodes and, optionally, hub nodes similar to many others.
// Node ids are offset so that they differ from dense indices.
func plantedEdges(seed int64, groups, groupSize int, pIn float64, noisePerNode, hubs, hubDegree int) (edges []SimEdge, groupOf map[int64]int) {
	const idBase = 1000
	r := rand.New(rand.NewSource(seed))
	n := groups * groupSize
	seen := make(map[[2]int]bool)
	add := func(a, b int, w float64) {
		if a > b {
			a, b = b, a
		}
		if a == b || seen[[2]int{a, b}] {
			return
		}
		seen[[2]int{a, b}] = true
		edges = append(edges, SimEdge{A: int64(a) + idBase, B: int64(b) + idBase, Weight: w})
	}
	groupOf = make(map[int64]int, n)
	for u := 0; u < n; u++ {
		groupOf[int64(u)+idBase] = u / groupSize
		for v := u + 1; v < (u/groupSize+1)*groupSize; v++ {
			if r.Float64() < pIn {
				add(u, v, 0.25+0.5*r.Float64())
			}
		}
	}
	for i := 0; i < n*noisePerNode; i++ {
		add(r.Intn(n), r.Intn(n), 0.2+0.1*r.Float64())
	}
	for h := 0; h < hubs; h++ {
		hub := r.Intn(n)
		for i := 0; i < hubDegree; i++ {
			add(hub, r.Intn(n), 0.3+0.3*r.Float64())
		}
	}
	return edges, groupOf
}

func testParams(k int) mhParams {
	return mhParams{
		numCommunities: k,
		maxPerNode:     4,
		maxCandidates:  7,
		alpha:          10,
		epsilon:        0.08,
		temperature:    1,
		seed:           1,
		workers:        4,
	}
}

// checkInvariants verifies that z and members describe the same assignment
// and that every node has between 1 and l distinct communities.
func checkInvariants(t *testing.T, s *mhState) {
	t.Helper()
	n, l, k := s.g.numNodes(), s.p.maxPerNode, s.p.numCommunities
	memberships := 0
	for u := int32(0); u < int32(n); u++ {
		row := s.row(u)
		comms := s.communities(u)
		if len(comms) < 1 || len(comms) > l {
			t.Fatalf("node %d has %d communities, want 1..%d", u, len(comms), l)
		}
		for i, c := range row {
			if i >= len(comms) {
				if c != -1 {
					t.Fatalf("node %d row %v is not packed", u, row)
				}
				continue
			}
			if c < 0 || int(c) >= k {
				t.Fatalf("node %d has community %d outside [0, %d)", u, c, k)
			}
			if slices.Contains(comms[:i], c) {
				t.Fatalf("node %d row %v has a duplicate community", u, row)
			}
			if !slices.Contains(s.members[c], u) {
				t.Fatalf("node %d is in community %d but not in its member list", u, c)
			}
		}
		memberships += len(comms)
	}
	listed := 0
	for c, m := range s.members {
		listed += len(m)
		sorted := slices.Clone(m)
		slices.Sort(sorted)
		if len(slices.Compact(sorted)) != len(m) {
			t.Fatalf("community %d lists a member twice", c)
		}
	}
	if listed != memberships {
		t.Fatalf("member lists hold %d entries, rows hold %d", listed, memberships)
	}
}

func TestInitializeRespectsCap(t *testing.T) {
	// Hubs are among the strongest neighbours of many seeds, which used to
	// put them in hundreds of communities.
	edges, _ := plantedEdges(1, 40, 25, 0.6, 3, 20, 300)
	g := newSimGraph(edges, 0, 0, 4)
	s := newMHState(g, testParams(g.numNodes()/4))
	checkInvariants(t, s)
}

func TestProposalDeltaMatchesObjective(t *testing.T) {
	edges, _ := plantedEdges(2, 20, 20, 0.6, 3, 5, 100)
	g := newSimGraph(edges, 0, 0, 1)
	p := testParams(g.numNodes() / 4)
	p.workers = 1
	p.batchSize = 1 // every proposal sees the exact current state
	s := newMHState(g, p)

	for epoch := 1; epoch <= 6; epoch++ {
		temperature := 1.0
		if epoch > 4 {
			temperature = 0 // greedy
		}
		before := s.objective().Score
		stats := s.runEpoch(epoch, temperature)
		after := s.objective().Score
		if stats.moved == 0 {
			t.Fatalf("epoch %d moved no node", epoch)
		}
		if diff := math.Abs(after - before - stats.gain); diff > 1e-6 {
			t.Fatalf("epoch %d: objective changed by %v but accepted deltas sum to %v", epoch, after-before, stats.gain)
		}
		if temperature == 0 && after < before {
			t.Fatalf("greedy epoch %d lowered the objective from %v to %v", epoch, before, after)
		}
		checkInvariants(t, s)
	}
}

func TestDetectCommunitiesIsDeterministic(t *testing.T) {
	edges, _ := plantedEdges(3, 30, 20, 0.6, 3, 5, 100)
	g := newSimGraph(edges, 0, 0, 4)
	run := func(seed uint64, workers int) []int32 {
		p := testParams(g.numNodes() / 4)
		p.seed, p.workers = seed, workers
		s := detectCommunities(g, p, 10, 1)
		checkInvariants(t, s)
		return s.z
	}
	base := run(1, 1)
	if !slices.Equal(base, run(1, 7)) {
		t.Error("the same seed gave different assignments with a different worker count")
	}
	if slices.Equal(base, run(2, 1)) {
		t.Error("different seeds gave identical assignments")
	}
}

func TestDetectCommunitiesRecoversPlantedGroups(t *testing.T) {
	const groups, groupSize = 40, 25
	edges, groupOf := plantedEdges(4, groups, groupSize, 0.6, 3, 0, 0)
	g := newSimGraph(edges, 0, 0, 4)
	initial := newMHState(g, testParams(2*groups)).objective().Score
	s := detectCommunities(g, testParams(2*groups), 60, 2)
	checkInvariants(t, s)
	if final := s.objective().Score; final <= initial {
		t.Errorf("objective did not improve: %v -> %v", initial, final)
	}

	// Count node pairs that share a community, split by whether they were
	// planted in the same group.
	n := int32(g.numNodes())
	share := func(a, b int32) bool {
		for _, c := range s.communities(a) {
			if slices.Contains(s.communities(b), c) {
				return true
			}
		}
		return false
	}
	var sameShared, sameTotal, shared int
	for a := int32(0); a < n; a++ {
		for b := a + 1; b < n; b++ {
			same := groupOf[g.ids[a]] == groupOf[g.ids[b]]
			sh := share(a, b)
			if same {
				sameTotal++
			}
			if sh {
				shared++
				if same {
					sameShared++
				}
			}
		}
	}
	recall := float64(sameShared) / float64(sameTotal)
	precision := float64(sameShared) / float64(shared)
	t.Logf("pairs sharing a community: recall %.3f, precision %.3f", recall, precision)
	if recall < 0.9 || precision < 0.9 {
		t.Errorf("planted groups not recovered: recall %.3f, precision %.3f, want both >= 0.9", recall, precision)
	}
}

// BenchmarkMHStep measures one node update on a 6000-node graph with hubs.
func BenchmarkMHStep(b *testing.B) {
	edges, _ := plantedEdges(1, 200, 30, 0.6, 3, 60, 800)
	g := newSimGraph(edges, 0, 0, 1)
	p := testParams(g.numNodes() / 4)
	p.workers = 1
	s := newMHState(g, p)
	n := g.numNodes()
	b.ReportAllocs()
	b.ResetTimer()
	for steps, epoch := 0, 1; steps < b.N; steps, epoch = steps+n, epoch+1 {
		s.runEpoch(epoch, p.temperature)
	}
}

func TestAssignmentsScores(t *testing.T) {
	edges, _ := plantedEdges(5, 10, 20, 0.7, 2, 0, 0)
	g := newSimGraph(edges, 0, 0, 2)
	s := detectCommunities(g, testParams(10), 20, 2)
	out := s.assignments()
	if len(out) != g.numNodes() {
		t.Fatalf("got assignments for %d nodes, want %d", len(out), g.numNodes())
	}
	for id, scores := range out {
		if len(scores) < 1 || len(scores) > s.p.maxPerNode {
			t.Fatalf("influencer %d has %d communities", id, len(scores))
		}
		for c, score := range scores {
			if score <= 0 || score > 1+1e-9 {
				t.Fatalf("influencer %d community %d has score %v outside (0, 1]", id, c, score)
			}
		}
	}
}

func TestNewSimGraphMaxNeighbors(t *testing.T) {
	// A star around node 1 plus a strong triangle 2-3-4.
	edges := []SimEdge{
		{1, 2, 0.30}, {1, 3, 0.25}, {1, 4, 0.20}, {1, 5, 0.15}, {1, 6, 0.10},
		{2, 3, 0.90}, {2, 4, 0.80}, {3, 4, 0.70},
		{7, 7, 1}, // self loops are ignored
	}
	degrees := func(g *simGraph) map[int64]int {
		d := make(map[int64]int)
		for u, id := range g.ids {
			nbrs, ws := g.neighbors(int32(u))
			d[id] = len(nbrs)
			for i, v := range nbrs {
				// Every edge must be present in both directions with the same weight.
				back, backW := g.neighbors(v)
				j := slices.Index(back, int32(u))
				if j < 0 || backW[j] != ws[i] {
					t.Fatalf("edge %d-%d is not symmetric", id, g.ids[v])
				}
				if i > 0 && ws[i] > ws[i-1] {
					t.Fatalf("row of %d is not sorted by descending weight", id)
				}
			}
		}
		return d
	}

	full := newSimGraph(edges, 0.15, 0, 2)
	if got, want := degrees(full), map[int64]int{1: 4, 2: 3, 3: 3, 4: 3, 5: 1}; !maps.Equal(got, want) || full.numEdges != 7 {
		t.Errorf("degrees with min weight 0.15: %v (%d edges), want %v (7 edges)", got, full.numEdges, want)
	}

	// With two neighbours each: node 1 keeps 2 and 3; nodes 2, 3 and 4 keep
	// the triangle; node 5 keeps its only edge, which is to node 1.
	sparse := newSimGraph(edges, 0.15, 2, 2)
	if got, want := degrees(sparse), map[int64]int{1: 3, 2: 3, 3: 3, 4: 2, 5: 1}; !maps.Equal(got, want) || sparse.numEdges != 6 {
		t.Errorf("degrees with 2 neighbours: %v (%d edges), want %v (6 edges)", got, sparse.numEdges, want)
	}
}
