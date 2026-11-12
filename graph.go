package simclusters

import (
	"slices"
	"sort"
)

// simGraph is the undirected influencer similarity graph in CSR form. Nodes
// are dense indices into ids, and each adjacency row is sorted by descending
// weight.
type simGraph struct {
	ids      []int64   // dense index -> influencer id, ascending
	off      []int64   // row offsets into nbr/w, len(ids)+1 entries
	nbr      []int32   // neighbour dense indices
	w        []float32 // edge weights, parallel to nbr
	rowW     []float64 // sum of edge weights per node
	numEdges int
}

func (g *simGraph) numNodes() int { return len(g.ids) }

func (g *simGraph) neighbors(u int32) ([]int32, []float32) {
	s, e := g.off[u], g.off[u+1]
	return g.nbr[s:e], g.w[s:e]
}

// newSimGraph builds the graph from edges with Weight >= minWeight. If
// maxNeighbors is positive the graph is then sparsified: an edge is kept only
// if it is among the maxNeighbors strongest edges of at least one of its
// endpoints (edges tied with the weakest of those are kept too). Nodes left
// without an edge are not part of the graph.
func newSimGraph(edges []SimEdge, minWeight float64, maxNeighbors, workers int) *simGraph {
	keep := func(e SimEdge) bool { return e.Weight >= minWeight && e.A != e.B }

	index := make(map[int64]int32)
	for _, e := range edges {
		if keep(e) {
			index[e.A] = 0
			index[e.B] = 0
		}
	}
	ids := make([]int64, 0, len(index))
	for id := range index {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for i, id := range ids {
		index[id] = int32(i)
	}

	// Endpoints as dense indices; -1 marks an edge that is not kept.
	ends := make([]int32, 2*len(edges))
	parallelRange(len(edges), workers, 1<<16, func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			if e := edges[i]; keep(e) {
				ends[2*i], ends[2*i+1] = index[e.A], index[e.B]
			} else {
				ends[2*i] = -1
			}
		}
	})

	n := len(ids)
	off := make([]int64, n+1)
	for i := range edges {
		if a := ends[2*i]; a >= 0 {
			off[a+1]++
			off[ends[2*i+1]+1]++
		}
	}
	for i := 0; i < n; i++ {
		off[i+1] += off[i]
	}
	nbr := make([]int32, off[n])
	w := make([]float32, off[n])
	cursor := slices.Clone(off[:n])
	for i, e := range edges {
		a := ends[2*i]
		if a < 0 {
			continue
		}
		b := ends[2*i+1]
		nbr[cursor[a]], w[cursor[a]] = b, float32(e.Weight)
		cursor[a]++
		nbr[cursor[b]], w[cursor[b]] = a, float32(e.Weight)
		cursor[b]++
	}
	parallelRange(n, workers, 64, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			sort.Sort(byWeightDesc{nbr[off[u]:off[u+1]], w[off[u]:off[u+1]]})
		}
	})

	if maxNeighbors > 0 {
		// weakest[u] is the weight of u's maxNeighbors-th strongest edge.
		weakest := make([]float32, n)
		for u := 0; u < n; u++ {
			if row := w[off[u]:off[u+1]]; len(row) > maxNeighbors {
				weakest[u] = row[maxNeighbors-1]
			}
		}
		// Compact the rows in place; the test is symmetric, so both
		// directions of an edge survive or go together.
		var out int64
		for u := 0; u < n; u++ {
			start, end := off[u], off[u+1]
			off[u] = out
			for i := start; i < end; i++ {
				if w[i] >= weakest[u] || w[i] >= weakest[nbr[i]] {
					nbr[out], w[out] = nbr[i], w[i]
					out++
				}
			}
		}
		off[n] = out
		nbr, w = nbr[:out], w[:out]
	}

	g := &simGraph{ids: ids, off: off, nbr: nbr, w: w, numEdges: len(nbr) / 2, rowW: make([]float64, n)}
	parallelRange(n, workers, 256, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			_, ws := g.neighbors(int32(u))
			var sum float64
			for _, x := range ws {
				sum += float64(x)
			}
			g.rowW[u] = sum
		}
	})
	return g
}

// byWeightDesc sorts one adjacency row by descending weight, breaking ties by
// neighbour index so the order is deterministic.
type byWeightDesc struct {
	nbr []int32
	w   []float32
}

func (r byWeightDesc) Len() int { return len(r.nbr) }
func (r byWeightDesc) Less(i, j int) bool {
	if r.w[i] != r.w[j] {
		return r.w[i] > r.w[j]
	}
	return r.nbr[i] < r.nbr[j]
}
func (r byWeightDesc) Swap(i, j int) {
	r.nbr[i], r.nbr[j] = r.nbr[j], r.nbr[i]
	r.w[i], r.w[j] = r.w[j], r.w[i]
}
