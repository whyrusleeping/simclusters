package simclusters

import (
	"log/slog"
	"math"
	"slices"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// cosineSimilarity is the cosine similarity of two sets viewed as binary
// vectors.
func cosineSimilarity(intersection, cardA, cardB float64) float64 {
	if cardA == 0 || cardB == 0 {
		return 0
	}
	return intersection / math.Sqrt(cardA*cardB)
}

// buildSimilarityEdges returns every pair of influencers whose follower sets
// have cosine similarity of at least threshold, sorted by (A, B).
//
// The result is exact. Rather than intersecting follower sets pair by pair,
// it counts co-follows: for influencer i, every follower contributes one to
// each other influencer they follow, which yields the intersection sizes of
// i with all other influencers at once. The total work is the sum over users
// of the squared number of influencers they follow.
func buildSimilarityEdges(gi *GraphIndex, influencers []int64, threshold float64, workers int, progress func(done int)) []SimEdge {
	start := time.Now()
	influencers = slices.Clone(influencers)
	slices.Sort(influencers)
	n := len(influencers)
	index := make(map[int64]int32, n)
	for i, id := range influencers {
		index[id] = int32(i)
	}

	// For every user, the influencers they follow, as ascending indices.
	var users []int64
	gi.Following.Range(func(u int64, _ *roaring64.Bitmap) bool {
		users = append(users, u)
		return true
	})
	const chunk = 4096
	type userList struct {
		user int64
		list []int32
	}
	chunks := make([][]userList, (len(users)+chunk-1)/chunk)
	parallelRange(len(chunks), workers, 1, func(_, lo, hi int) {
		for c := lo; c < hi; c++ {
			var lists []userList
			var all []int32 // one backing array per chunk
			var lens []int
			for _, u := range users[c*chunk : min((c+1)*chunk, len(users))] {
				before := len(all)
				it := gi.GetFollowing(u).Iterator()
				for it.HasNext() {
					if i, ok := index[int64(it.Next())]; ok {
						all = append(all, i)
					}
				}
				if len(all) > before {
					lists = append(lists, userList{user: u})
					lens = append(lens, len(all)-before)
				}
			}
			for i, off := 0, 0; i < len(lists); i++ {
				lists[i].list = all[off : off+lens[i] : off+lens[i]]
				off += lens[i]
			}
			chunks[c] = lists
		}
	})
	follows := make(map[int64][]int32, len(users))
	var coFollows float64
	for _, lists := range chunks {
		for _, ul := range lists {
			follows[ul.user] = ul.list
			coFollows += float64(len(ul.list)) * float64(len(ul.list)-1) / 2
		}
	}
	slog.Info("influencer follow lists built", "users", len(follows), "co_follows", coFollows,
		"took", time.Since(start).Round(time.Millisecond))

	followers := make([]*roaring64.Bitmap, n)
	cards := make([]float64, n)
	for i, id := range influencers {
		followers[i] = gi.GetFollowers(id)
		cards[i] = float64(followers[i].GetCardinality())
	}

	rows := make([][]SimEdge, n)
	counters := make([][]uint32, max(workers, 1))
	// Slightly loosened so that rounding in the cheap pre-check can never
	// reject a pair that the exact comparison below would keep.
	loose := threshold * threshold * (1 - 1e-9)
	parallelRange(n, workers, 1, func(worker, lo, hi int) {
		if counters[worker] == nil {
			counters[worker] = make([]uint32, n)
		}
		counts := counters[worker]
		var buf [512]uint64
		for i := lo; i < hi; i++ {
			// counts[j] = number of followers of i that also follow j, for j > i.
			it := followers[i].ManyIterator()
			for m := it.NextMany(buf[:]); m > 0; m = it.NextMany(buf[:]) {
				for _, u := range buf[:m] {
					list := follows[int64(u)]
					for k := len(list) - 1; k >= 0 && int(list[k]) > i; k-- {
						counts[list[k]]++
					}
				}
			}
			var row []SimEdge
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
					row = append(row, SimEdge{A: influencers[i], B: influencers[j], Weight: sim})
				}
			}
			rows[i] = row
			if progress != nil {
				progress(1)
			}
		}
	})

	total := 0
	for _, row := range rows {
		total += len(row)
	}
	edges := make([]SimEdge, 0, total)
	for i, row := range rows {
		edges = append(edges, row...)
		rows[i] = nil
	}
	slog.Info("similarity graph built", "influencers", n, "edges", len(edges), "threshold", threshold,
		"took", time.Since(start).Round(time.Millisecond))
	return edges
}
