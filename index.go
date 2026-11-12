package simclusters

import (
	"math"
	"math/bits"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
)

// GraphIndex holds the follow graph in compressed sparse rows. Users are
// numbered densely in ascending id order, and each user's row lists the users
// they follow: ascending dense indices, without duplicates. Only this
// direction is stored; the number of distinct followers of every user is kept
// alongside.
//
// AddEdgesBatch may be called from many goroutines at once. Added edges are
// buffered and folded into the rows by the first read after them; reads must
// not overlap with AddEdgesBatch.
type GraphIndex struct {
	mu      sync.Mutex
	pending []edgeBlock

	ids   []int64  // dense index -> user id, ascending
	off   []uint64 // row offsets into adj; len(ids)+1 entries once built
	adj   []uint32 // dense indices of followed users
	inDeg []uint32 // number of distinct followers per user
}

// edgeBlock is a batch of buffered edges. pairs holds (from, to) interleaved;
// wide is used instead when an id does not fit in 32 bits.
type edgeBlock struct {
	pairs []uint32
	maxID uint32 // largest id in pairs
	wide  []Edge
}

func newEdgeBlock(edges []Edge) edgeBlock {
	for _, e := range edges {
		if uint64(e.From) > math.MaxUint32 || uint64(e.To) > math.MaxUint32 {
			return edgeBlock{wide: slices.Clone(edges)}
		}
	}
	b := edgeBlock{pairs: make([]uint32, 2*len(edges))}
	for i, e := range edges {
		b.pairs[2*i], b.pairs[2*i+1] = uint32(e.From), uint32(e.To)
		b.maxID = max(b.maxID, uint32(e.From), uint32(e.To))
	}
	return b
}

// NewGraphIndex creates an empty graph index.
func NewGraphIndex() *GraphIndex {
	return &GraphIndex{}
}

// AddEdgesBatch adds follow edges. Duplicate edges are harmless. The slice is
// neither retained nor modified.
func (gi *GraphIndex) AddEdgesBatch(edges []Edge) {
	if len(edges) == 0 {
		return
	}
	b := newEdgeBlock(edges)
	gi.mu.Lock()
	gi.pending = append(gi.pending, b)
	gi.mu.Unlock()
}

// NumUsers returns the number of users that follow or are followed by anyone.
func (gi *GraphIndex) NumUsers() int {
	gi.build(runtime.GOMAXPROCS(0))
	return len(gi.ids)
}

// Following returns the ids of the users that userID follows, ascending.
func (gi *GraphIndex) Following(userID int64) []int64 {
	gi.build(runtime.GOMAXPROCS(0))
	u, ok := gi.index(userID)
	if !ok {
		return nil
	}
	row := gi.row(u)
	out := make([]int64, len(row))
	for i, v := range row {
		out[i] = gi.ids[v]
	}
	return out
}

// FollowerCount returns the number of distinct followers of userID.
func (gi *GraphIndex) FollowerCount(userID int64) int {
	gi.build(runtime.GOMAXPROCS(0))
	if u, ok := gi.index(userID); ok {
		return int(gi.inDeg[u])
	}
	return 0
}

// index returns the dense index of a user id. The index must be built.
func (gi *GraphIndex) index(id int64) (uint32, bool) {
	i, ok := slices.BinarySearch(gi.ids, id)
	return uint32(i), ok
}

// row returns the users that dense user u follows. The index must be built.
func (gi *GraphIndex) row(u uint32) []uint32 {
	return gi.adj[gi.off[u]:gi.off[u+1]]
}

// build folds pending edges into the rows.
func (gi *GraphIndex) build(workers int) {
	gi.mu.Lock()
	defer gi.mu.Unlock()
	if len(gi.pending) == 0 {
		return
	}
	blocks := gi.pending
	gi.pending = nil
	if len(gi.ids) > 0 {
		// Rare (only when edges are added after a read): rebuild with the
		// existing edges as one more block.
		var existing []Edge
		for u := range gi.ids {
			for _, v := range gi.row(uint32(u)) {
				existing = append(existing, Edge{From: gi.ids[u], To: gi.ids[v]})
			}
		}
		blocks = append(blocks, newEdgeBlock(existing))
	}
	workers = max(workers, 1)

	gi.ids = resolveIDs(blocks, workers)
	n := len(gi.ids)

	// Out-degrees including duplicates. Rows are usually contiguous in the
	// input, so a run of edges from one user takes a single atomic add.
	deg := make([]uint32, n)
	forRuns := func(fn func(from uint32, run []uint32)) {
		parallelRange(len(blocks), workers, 1, func(_, lo, hi int) {
			for _, b := range blocks[lo:hi] {
				p := b.pairs
				for i := 0; i < len(p); {
					j := i + 2
					for j < len(p) && p[j] == p[i] {
						j += 2
					}
					fn(p[i], p[i:j])
					i = j
				}
			}
		})
	}
	forRuns(func(from uint32, run []uint32) {
		atomic.AddUint32(&deg[from], uint32(len(run)/2))
	})
	off := make([]uint64, n+1)
	for u := 0; u < n; u++ {
		off[u+1] = off[u] + uint64(deg[u])
	}
	cursor := slices.Clone(off[:n])
	adj := make([]uint32, off[n])
	forRuns(func(from uint32, run []uint32) {
		k := uint64(len(run) / 2)
		pos := atomic.AddUint64(&cursor[from], k) - k
		for i := 1; i < len(run); i += 2 {
			adj[pos] = run[i]
			pos++
		}
	})
	cursor = nil
	clear(blocks) // let the buffered edges be collected

	// Sort and deduplicate each row; deg becomes the deduplicated length.
	var dups atomic.Int64
	parallelRange(n, workers, 1024, func(_, lo, hi int) {
		var removed int64
		for u := lo; u < hi; u++ {
			row := adj[off[u]:off[u+1]]
			if !slices.IsSorted(row) {
				slices.Sort(row)
			}
			k := len(slices.Compact(row))
			removed += int64(len(row) - k)
			deg[u] = uint32(k)
		}
		dups.Add(removed)
	})
	if dups.Load() > 0 {
		var out uint64
		for u := 0; u < n; u++ {
			start := off[u]
			off[u] = out
			copy(adj[out:], adj[start:start+uint64(deg[u])])
			out += uint64(deg[u])
		}
		off[n] = out
		if out < uint64(len(adj))/8*7 {
			adj = slices.Clone(adj[:out])
		} else {
			adj = adj[:out:out]
		}
	}
	gi.off, gi.adj = off, adj
	gi.inDeg = countInDegrees(n, adj, workers)
	// The released edge buffers are about twice the size of the rows. Collect
	// them now; otherwise the heap may grow to twice its peak before the next
	// collection does.
	runtime.GC()
}

// resolveIDs returns the distinct user ids of the blocks in ascending order
// and rewrites every block to pairs of dense indices into them.
func resolveIDs(blocks []edgeBlock, workers int) []int64 {
	narrow := true
	var maxID uint32
	for _, b := range blocks {
		narrow = narrow && b.wide == nil
		maxID = max(maxID, b.maxID)
	}

	if narrow {
		// Mark the ids in a bitmap; a dense index is then the rank of its bit.
		words := make([]uint64, uint64(maxID)/64+1)
		parallelRange(len(blocks), workers, 1, func(_, lo, hi int) {
			for _, b := range blocks[lo:hi] {
				for _, id := range b.pairs {
					w, bit := &words[id/64], uint64(1)<<(id%64)
					// Popular ids recur often; reading first avoids
					// contended writes to their word.
					if atomic.LoadUint64(w)&bit == 0 {
						atomic.OrUint64(w, bit)
					}
				}
			}
		})
		rank := make([]uint32, len(words))
		var ids []int64
		for i, w := range words {
			rank[i] = uint32(len(ids))
			for ; w != 0; w &= w - 1 {
				ids = append(ids, int64(i*64+bits.TrailingZeros64(w)))
			}
		}
		parallelRange(len(blocks), workers, 1, func(_, lo, hi int) {
			for _, b := range blocks[lo:hi] {
				for i, id := range b.pairs {
					b.pairs[i] = rank[id/64] + uint32(bits.OnesCount64(words[id/64]&(uint64(1)<<(id%64)-1)))
				}
			}
		})
		return ids
	}

	// Some id does not fit in 32 bits: sort the distinct ids instead.
	uniq := make([][]int64, len(blocks))
	parallelRange(len(blocks), workers, 1, func(_, lo, hi int) {
		for k := lo; k < hi; k++ {
			b := blocks[k]
			var ids []int64
			if b.wide != nil {
				ids = make([]int64, 0, 2*len(b.wide))
				for _, e := range b.wide {
					ids = append(ids, e.From, e.To)
				}
			} else {
				ids = make([]int64, len(b.pairs))
				for i, id := range b.pairs {
					ids[i] = int64(id)
				}
			}
			slices.Sort(ids)
			uniq[k] = slices.Clip(slices.Compact(ids))
		}
	})
	ids := slices.Concat(uniq...)
	uniq = nil
	slices.Sort(ids)
	ids = slices.Clip(slices.Compact(ids))
	if len(ids) > math.MaxUint32 {
		panic("simclusters: more than 2^32 distinct users")
	}
	dense := func(id int64) uint32 {
		i, _ := slices.BinarySearch(ids, id)
		return uint32(i)
	}
	parallelRange(len(blocks), workers, 1, func(_, lo, hi int) {
		for k := lo; k < hi; k++ {
			b := &blocks[k]
			if b.wide != nil {
				b.pairs = make([]uint32, 2*len(b.wide))
				for i, e := range b.wide {
					b.pairs[2*i], b.pairs[2*i+1] = dense(e.From), dense(e.To)
				}
				b.wide = nil
				continue
			}
			for i, id := range b.pairs {
				b.pairs[i] = dense(int64(id))
			}
		}
	})
	return ids
}

// countInDegrees counts how often each of n users appears in adj. Workers
// count into private arrays, so popular users cause no write contention.
func countInDegrees(n int, adj []uint32, workers int) []uint32 {
	parts := min(workers, 16, max(len(adj)/max(n, 1), 1))
	partial := make([][]uint32, parts)
	parallelRange(len(adj), parts, (len(adj)+parts-1)/parts, func(w, lo, hi int) {
		if partial[w] == nil {
			partial[w] = make([]uint32, n)
		}
		counts := partial[w]
		for _, v := range adj[lo:hi] {
			counts[v]++
		}
	})
	// Chunks are handed out dynamically, so any worker may have had none.
	partial = slices.DeleteFunc(partial, func(p []uint32) bool { return p == nil })
	if len(partial) == 0 {
		return make([]uint32, n)
	}
	inDeg := partial[0]
	parallelRange(n, workers, 1<<16, func(_, lo, hi int) {
		for _, p := range partial[1:] {
			for u := lo; u < hi; u++ {
				inDeg[u] += p[u]
			}
		}
	})
	return inDeg
}
