package simclusters

import (
	"cmp"
	"slices"
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	smap "github.com/zolstein/sync-map"
)

// numLocks is the number of lock stripes guarding bitmap updates per
// direction.
const numLocks = 1024

// GraphIndex holds the follow graph in both directions.
//
// AddEdgesBatch may be called from many goroutines at once. Reading bitmaps
// is only safe once ingestion has finished.
type GraphIndex struct {
	// Following maps a user to the users they follow.
	Following smap.Map[int64, *roaring64.Bitmap]
	// Followers maps a user to their followers.
	Followers smap.Map[int64, *roaring64.Bitmap]

	followingLocks [numLocks]sync.Mutex
	followersLocks [numLocks]sync.Mutex
}

// NewGraphIndex creates an empty graph index.
func NewGraphIndex() *GraphIndex {
	return &GraphIndex{}
}

// AddEdgesBatch adds follow edges to both directions of the index. Duplicate
// edges are harmless. The slice is not retained, but it is reordered.
func (gi *GraphIndex) AddEdgesBatch(edges []Edge) {
	var buf []uint64

	slices.SortFunc(edges, func(a, b Edge) int { return cmp.Compare(a.From, b.From) })
	for i := 0; i < len(edges); {
		from := edges[i].From
		buf = buf[:0]
		for ; i < len(edges) && edges[i].From == from; i++ {
			buf = append(buf, uint64(edges[i].To))
		}
		addMany(&gi.Following, &gi.followingLocks, from, buf)
	}

	slices.SortFunc(edges, func(a, b Edge) int { return cmp.Compare(a.To, b.To) })
	for i := 0; i < len(edges); {
		to := edges[i].To
		buf = buf[:0]
		for ; i < len(edges) && edges[i].To == to; i++ {
			buf = append(buf, uint64(edges[i].From))
		}
		addMany(&gi.Followers, &gi.followersLocks, to, buf)
	}
}

func addMany(m *smap.Map[int64, *roaring64.Bitmap], locks *[numLocks]sync.Mutex, key int64, vals []uint64) {
	bm, ok := m.Load(key)
	if !ok {
		// LoadOrStore, so that two goroutines creating the same key agree
		// on one bitmap.
		bm, _ = m.LoadOrStore(key, roaring64.New())
	}
	lk := &locks[uint64(key)%numLocks]
	lk.Lock()
	bm.AddMany(vals)
	lk.Unlock()
}

// GetFollowing returns the users that userID follows. The result must not be
// modified.
func (gi *GraphIndex) GetFollowing(userID int64) *roaring64.Bitmap {
	if bm, ok := gi.Following.Load(userID); ok {
		return bm
	}
	return roaring64.New()
}

// GetFollowers returns the followers of userID. The result must not be
// modified.
func (gi *GraphIndex) GetFollowers(userID int64) *roaring64.Bitmap {
	if bm, ok := gi.Followers.Load(userID); ok {
		return bm
	}
	return roaring64.New()
}

// FollowerCount returns the number of distinct followers of userID.
func (gi *GraphIndex) FollowerCount(userID int64) int {
	if bm, ok := gi.Followers.Load(userID); ok {
		return int(bm.GetCardinality())
	}
	return 0
}
