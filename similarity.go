package simclusters

import (
	"cmp"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"math"
	"slices"
	"time"
)

// cosineSimilarity is the cosine similarity of two sets viewed as binary
// vectors.
func cosineSimilarity(intersection, cardA, cardB float64) float64 {
	if cardA == 0 || cardB == 0 {
		return 0
	}
	return intersection / math.Sqrt(cardA*cardB)
}

// simMatrix is the upper triangle of the influencer similarity matrix: row i
// holds the influencers j > i similar to influencer i, ascending.
type simMatrix struct {
	ids  []int64     // influencer ids, ascending
	cols [][]int32   // cols[i]: indices j > i, ascending
	ws   [][]float64 // similarities, parallel to cols
}

func (m *simMatrix) numEdges() int {
	total := 0
	for _, row := range m.cols {
		total += len(row)
	}
	return total
}

// edges yields every pair once, ordered by (A, B).
func (m *simMatrix) edges() iter.Seq[SimEdge] {
	return func(yield func(SimEdge) bool) {
		for i, row := range m.cols {
			for k, j := range row {
				if !yield(SimEdge{A: m.ids[i], B: m.ids[j], Weight: m.ws[i][k]}) {
					return
				}
			}
		}
	}
}

// newSimMatrix builds a matrix over the influencer ids (sorted, without
// duplicates) from edges strictly ordered by (A, B) with A < B, both of them
// influencers.
func newSimMatrix(ids []int64, edges iter.Seq[SimEdge]) (*simMatrix, error) {
	if !slices.IsSortedFunc(ids, cmp.Compare) || len(slices.Compact(slices.Clone(ids))) != len(ids) {
		return nil, errors.New("influencer ids are not sorted and distinct")
	}
	m := &simMatrix{ids: ids, cols: make([][]int32, len(ids)), ws: make([][]float64, len(ids))}
	var cols []int32
	var ws []float64
	row := -1
	flush := func() {
		if row >= 0 && len(cols) > 0 {
			m.cols[row], m.ws[row] = slices.Clone(cols), slices.Clone(ws)
		}
		cols, ws = cols[:0], ws[:0]
	}
	errOrder := func(e SimEdge) error {
		return fmt.Errorf("edge %d-%d: edges are not strictly ordered by (A, B) with A < B", e.A, e.B)
	}
	errUnknown := func(e SimEdge) error {
		return fmt.Errorf("edge %d-%d: endpoint is not an influencer", e.A, e.B)
	}
	for e := range edges {
		a := row
		if a < 0 || e.A != ids[a] {
			var ok bool
			if a, ok = slices.BinarySearch(ids, e.A); !ok {
				return nil, errUnknown(e)
			}
			if a < row {
				return nil, errOrder(e)
			}
			flush()
			row = a
		}
		// Within a row B ascends, so search forward from the previous column.
		lo := a + 1
		if len(cols) > 0 {
			lo = int(cols[len(cols)-1]) + 1
		}
		b, ok := gallopSearch(ids, lo, e.B)
		if !ok {
			if _, known := slices.BinarySearch(ids, e.B); known {
				return nil, errOrder(e)
			}
			return nil, errUnknown(e)
		}
		cols, ws = append(cols, int32(b)), append(ws, e.Weight)
	}
	flush()
	return m, nil
}

// gallopSearch finds target in ids[lo:] by exponential then binary search,
// which is fast when it lies close to lo. It returns the insertion index.
func gallopSearch(ids []int64, lo int, target int64) (int, bool) {
	if lo >= len(ids) {
		return len(ids), false
	}
	step, hi := 1, lo
	for hi < len(ids) && ids[hi] < target {
		lo = hi + 1
		hi += step
		step *= 2
	}
	hi = min(hi+1, len(ids))
	i, ok := slices.BinarySearch(ids[lo:hi], target)
	return lo + i, ok
}

// buildSimilarity returns every pair of influencers whose follower sets have
// cosine similarity of at least threshold.
//
// The result is exact. Rather than intersecting follower sets pair by pair,
// it counts co-follows: for influencer i, every follower contributes one to
// each other influencer they follow, which yields the intersection sizes of
// i with all other influencers at once. The total work is the sum over users
// of the squared number of influencers they follow.
func buildSimilarity(gi *GraphIndex, influencers []int64, threshold float64, workers int, progress func(done int)) *simMatrix {
	start := time.Now()
	gi.build(workers)
	ids := slices.Clone(influencers)
	slices.Sort(ids)
	ids = slices.Compact(ids)
	n, numUsers := len(ids), len(gi.ids)

	infOf := make([]int32, numUsers) // dense user -> influencer index, or -1
	parallelRange(numUsers, workers, 1<<16, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			infOf[u] = -1
		}
	})
	cards := make([]float64, n)
	for i, id := range ids {
		if u, ok := gi.index(id); ok {
			infOf[u] = int32(i)
			cards[i] = float64(gi.inDeg[u])
		}
	}

	// For every user following at least two influencers (no one else
	// contributes a co-follow), those influencers as ascending indices.
	listOff := make([]uint64, numUsers+1)
	parallelRange(numUsers, workers, 4096, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			c := 0
			for _, v := range gi.row(uint32(u)) {
				if infOf[v] >= 0 {
					c++
				}
			}
			if c >= 2 {
				listOff[u+1] = uint64(c)
			}
		}
	})
	var coFollows float64
	for u := 0; u < numUsers; u++ {
		c := float64(listOff[u+1])
		coFollows += c * (c - 1) / 2
		listOff[u+1] += listOff[u]
	}
	lists := make([]int32, listOff[numUsers])
	parallelRange(numUsers, workers, 4096, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			if listOff[u] == listOff[u+1] {
				continue
			}
			pos := listOff[u]
			for _, v := range gi.row(uint32(u)) {
				if i := infOf[v]; i >= 0 {
					lists[pos] = i
					pos++
				}
			}
		}
	})
	infOf = nil

	// The transpose: for every influencer, those of its followers that have
	// a list. Users are split into blocks of similar list volume; each block
	// writes to its own reserved range of every influencer's row, so rows
	// come out in user order without atomics.
	blocks := max(min(workers, 64), 1)
	bounds := make([]int, blocks+1)
	for b := 1; b < blocks; b++ {
		target := listOff[numUsers] / uint64(blocks) * uint64(b)
		bounds[b], _ = slices.BinarySearch(listOff[:numUsers+1], target)
		bounds[b] = min(max(bounds[b], bounds[b-1]), numUsers)
	}
	bounds[blocks] = numUsers
	cursor := make([]uint64, blocks*n) // cursor[b*n+i]
	parallelRange(blocks, workers, 1, func(_, lo, hi int) {
		for b := lo; b < hi; b++ {
			c := cursor[b*n : (b+1)*n]
			for _, i := range lists[listOff[bounds[b]]:listOff[bounds[b+1]]] {
				c[i]++
			}
		}
	})
	folOff := make([]uint64, n+1)
	for i, pos := 0, uint64(0); i < n; i++ {
		for b := 0; b < blocks; b++ {
			c := cursor[b*n+i]
			cursor[b*n+i] = pos
			pos += c
		}
		folOff[i+1] = pos
	}
	followers := make([]uint32, folOff[n])
	parallelRange(blocks, workers, 1, func(_, lo, hi int) {
		for b := lo; b < hi; b++ {
			c := cursor[b*n : (b+1)*n]
			for u := bounds[b]; u < bounds[b+1]; u++ {
				for _, i := range lists[listOff[u]:listOff[u+1]] {
					followers[c[i]] = uint32(u)
					c[i]++
				}
			}
		}
	})
	cursor = nil
	slog.Info("influencer follow lists built", "users", numUsers, "co_follows", coFollows,
		"took", time.Since(start).Round(time.Millisecond))

	m := &simMatrix{ids: ids, cols: make([][]int32, n), ws: make([][]float64, n)}
	type scratch struct {
		counts []uint32
		cols   []int32
		ws     []float64
	}
	scratches := make([]*scratch, max(workers, 1))
	// Slightly loosened so that rounding in the cheap pre-check can never
	// reject a pair that the exact comparison below would keep.
	loose := threshold * threshold * (1 - 1e-9)
	parallelRange(n, workers, 1, func(worker, lo, hi int) {
		if scratches[worker] == nil {
			scratches[worker] = &scratch{counts: make([]uint32, n)}
		}
		sc := scratches[worker]
		counts := sc.counts
		for i := lo; i < hi; i++ {
			// counts[j] = number of followers of i that also follow j, for j > i.
			for _, u := range followers[folOff[i]:folOff[i+1]] {
				list := lists[listOff[u]:listOff[u+1]]
				for k := len(list) - 1; k >= 0 && int(list[k]) > i; k-- {
					counts[list[k]]++
				}
			}
			sc.cols, sc.ws = sc.cols[:0], sc.ws[:0]
			limit := loose * cards[i]
			for j := i + 1; j < n; j++ {
				c := float64(counts[j])
				if c == 0 {
					continue
				}
				counts[j] = 0
				if c*c < limit*cards[j] {
					continue
				}
				if sim := cosineSimilarity(c, cards[i], cards[j]); sim >= threshold {
					sc.cols = append(sc.cols, int32(j))
					sc.ws = append(sc.ws, sim)
				}
			}
			if len(sc.cols) > 0 {
				m.cols[i], m.ws[i] = slices.Clone(sc.cols), slices.Clone(sc.ws)
			}
			if progress != nil {
				progress(1)
			}
		}
	})
	slog.Info("similarity graph built", "influencers", n, "edges", m.numEdges(), "threshold", threshold,
		"took", time.Since(start).Round(time.Millisecond))
	return m
}
