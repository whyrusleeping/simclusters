package simclusters

import (
	"sort"
	"sync/atomic"
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

// newSimGraph builds the graph from the matrix entries with weight >=
// minWeight. If maxNeighbors is positive the graph is then sparsified: an
// edge is kept only if it is among the maxNeighbors strongest edges of at
// least one of its endpoints (edges tied with the weakest of those are kept
// too). Influencers left without an edge are not part of the graph.
func newSimGraph(m *simMatrix, minWeight float64, maxNeighbors, workers int) *simGraph {
	numInf := len(m.ids)
	deg := make([]int64, numInf)
	parallelRange(numInf, workers, 64, func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			for k, j := range m.cols[i] {
				if m.ws[i][k] >= minWeight {
					atomic.AddInt64(&deg[i], 1)
					atomic.AddInt64(&deg[j], 1)
				}
			}
		}
	})

	// Nodes are the influencers with an edge, in influencer order.
	node := make([]int32, numInf)
	var ids []int64
	for i, d := range deg {
		node[i] = -1
		if d > 0 {
			node[i] = int32(len(ids))
			ids = append(ids, m.ids[i])
		}
	}
	n := len(ids)
	off := make([]int64, n+1)
	for i, d := range deg {
		if node[i] >= 0 {
			off[node[i]+1] = off[node[i]] + d
		}
	}
	cursor := make([]int64, n)
	copy(cursor, off[:n])
	nbr := make([]int32, off[n])
	w := make([]float32, off[n])
	parallelRange(numInf, workers, 64, func(_, lo, hi int) {
		for i := lo; i < hi; i++ {
			a := node[i]
			for k, j := range m.cols[i] {
				if m.ws[i][k] < minWeight {
					continue
				}
				b, wt := node[j], float32(m.ws[i][k])
				p := atomic.AddInt64(&cursor[a], 1) - 1
				nbr[p], w[p] = b, wt
				p = atomic.AddInt64(&cursor[b], 1) - 1
				nbr[p], w[p] = a, wt
			}
		}
	})
	// Fill order varies between runs; sorting by (weight, neighbour) makes
	// every row deterministic.
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
		// The test is symmetric, so both directions of an edge survive or
		// go together.
		keep := func(u int, i int64) bool { return w[i] >= weakest[u] || w[i] >= weakest[nbr[i]] }
		kept := make([]int64, n+1)
		parallelRange(n, workers, 256, func(_, lo, hi int) {
			for u := lo; u < hi; u++ {
				for i := off[u]; i < off[u+1]; i++ {
					if keep(u, i) {
						kept[u+1]++
					}
				}
			}
		})
		for u := 0; u < n; u++ {
			kept[u+1] += kept[u]
		}
		sparseNbr := make([]int32, kept[n])
		sparseW := make([]float32, kept[n])
		parallelRange(n, workers, 256, func(_, lo, hi int) {
			for u := lo; u < hi; u++ {
				out := kept[u]
				for i := off[u]; i < off[u+1]; i++ {
					if keep(u, i) {
						sparseNbr[out], sparseW[out] = nbr[i], w[i]
						out++
					}
				}
			}
		})
		off, nbr, w = kept, sparseNbr, sparseW
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
