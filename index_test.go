package simclusters

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestGraphIndexBuild(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	randomEdges := func(n int, id func() int64) []Edge {
		edges := make([]Edge, n)
		for i := range edges {
			edges[i] = Edge{From: id(), To: id()}
		}
		// Duplicates, and runs of one follower as in a sorted export.
		edges = append(edges, edges[:n/10]...)
		for i := 0; i < 50; i++ {
			edges = append(edges, Edge{From: 7, To: id()})
		}
		return edges
	}
	small := func() int64 { return r.Int63n(300) }
	for name, batches := range map[string][][]Edge{
		"32-bit ids":    {randomEdges(3000, small), randomEdges(2000, small)},
		"largest ids":   {randomEdges(500, small), {{From: math.MaxUint32, To: 0}, {From: math.MaxInt64, To: math.MaxUint32 + 1}}},
		"negative ids":  {randomEdges(500, func() int64 { return r.Int63n(300) - 150 })},
		"mixed batches": {randomEdges(3000, small), {{From: 1 << 40, To: 3}, {From: 3, To: 1 << 40}}, randomEdges(100, small)},
	} {
		t.Run(name, func(t *testing.T) {
			var all []Edge
			var built *GraphIndex
			for _, workers := range []int{1, 4} {
				gi := NewGraphIndex()
				all = all[:0]
				for _, b := range batches {
					gi.AddEdgesBatch(b)
					all = append(all, b...)
				}
				gi.build(workers)
				ingestCheckGraph(t, gi, all)
				if built != nil && (!slices.Equal(gi.ids, built.ids) || !slices.Equal(gi.off, built.off) ||
					!slices.Equal(gi.adj, built.adj) || !slices.Equal(gi.inDeg, built.inDeg)) {
					t.Fatal("index depends on the number of workers")
				}
				built = gi
			}

			// Edges added after a read are merged with the existing ones.
			more := []Edge{{From: 7, To: 123456}, {From: 99999, To: 7}, all[0]}
			built.AddEdgesBatch(more)
			ingestCheckGraph(t, built, append(all, more...))
		})
	}
}
