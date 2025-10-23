package simclusters

import (
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	"github.com/schollz/progressbar/v3"
	smap "github.com/zolstein/sync-map"
)

// GraphIndex maintains efficient indices for the follow graph
type GraphIndex struct {
	mu sync.RWMutex

	// Forward index: follower -> list of users they follow
	Following smap.Map[int64, *roaring64.Bitmap]
	//Following    map[int64]*roaring64.Bitmap
	followingLks []sync.Mutex

	// Inverted index: followed (influencer) -> list of followers
	//Followers    map[int64]*roaring64.Bitmap
	Followers    smap.Map[int64, *roaring64.Bitmap]
	followersLks []sync.Mutex

	// User metadata
	Users map[int64]*User

	// Influencer tracking
	Influencers map[int64]bool
}

const numLocks = 100

// NewGraphIndex creates a new graph index
func NewGraphIndex() *GraphIndex {
	return &GraphIndex{
		//Following:        make(map[int64]*roaring64.Bitmap),
		//Followers:        make(map[int64]*roaring64.Bitmap),
		followingLks: make([]sync.Mutex, numLocks),
		followersLks: make([]sync.Mutex, numLocks),
		Users:        make(map[int64]*User),
		Influencers:  make(map[int64]bool),
	}
}

// AddEdge adds a follow edge to the index

/*
func (gi *GraphIndex) AddEdge(from, to int64) {
	// Add to forward index
	gi.followingLks[from%numLocks].Lock()
	if _, exists := gi.Following[from]; !exists {
		gi.Following[from] = roaring64.New()
	}
	gi.Following[from].Add(uint64(to))
	gi.followingLks[from%numLocks].Unlock()

	// Add to inverted index
	gi.followersLks[to%numLocks].Lock()
	if _, exists := gi.Followers[to]; !exists {
		gi.Followers[to] = roaring64.New()
	}
	gi.Followers[to].Add(uint64(from))
	gi.followersLks[to%numLocks].Unlock()

	gi.mu.Lock()
	defer gi.mu.Unlock()
	// Ensure users exist
	if _, exists := gi.Users[from]; !exists {
		gi.Users[from] = &User{
			ID:            from,
			FollowerCount: 0,
			IsInfluencer:  false,
		}
	}

	if touser, exists := gi.Users[to]; !exists {
		gi.Users[to] = &User{
			ID:            to,
			FollowerCount: 1,
			IsInfluencer:  false,
		}
	} else {
		touser.FollowerCount++
	}
}
*/

// This uses a lot more memory
func (gi *GraphIndex) AddEdgesBatch(edges []Edge) {
	following := make(map[int64]*roaring64.Bitmap)
	followers := make(map[int64]*roaring64.Bitmap)

	// Add to forward index
	for _, e := range edges {
		fmap, ok := following[e.From]
		if !ok {
			fmap = roaring64.New()
			following[e.From] = fmap
		}
		fmap.Add(uint64(e.To))
	}

	// Add to inverted index
	for _, e := range edges {
		fmap, ok := followers[e.To]
		if !ok {
			fmap = roaring64.New()
			followers[e.To] = fmap
		}
		fmap.Add(uint64(e.From))
	}

	for k, v := range following {
		fmap, ok := gi.Following.Load(k)
		if !ok {
			fmap = roaring64.New()
			gi.Following.Store(k, fmap)
		}
		gi.followingLks[k%numLocks].Lock()
		fmap.Or(v)
		gi.followingLks[k%numLocks].Unlock()
	}

	for k, v := range followers {
		fmap, ok := gi.Followers.Load(k)
		if !ok {
			fmap = roaring64.New()
			gi.Followers.Store(k, fmap)
		}
		gi.followersLks[k%numLocks].Lock()
		fmap.Or(v)
		gi.followersLks[k%numLocks].Unlock()
	}

	gi.mu.Lock()
	defer gi.mu.Unlock()
	// Ensure users exist
	for _, e := range edges {
		if _, exists := gi.Users[e.From]; !exists {
			gi.Users[e.From] = &User{
				ID:            e.From,
				FollowerCount: 0,
				IsInfluencer:  false,
			}
		}

		if touser, exists := gi.Users[e.To]; !exists {
			gi.Users[e.To] = &User{
				ID:            e.To,
				FollowerCount: 1,
				IsInfluencer:  false,
			}
		} else {
			touser.FollowerCount++
		}
	}
}

func (gi *GraphIndex) AddEdges(edges []Edge) {
	// Add to forward index
	for _, e := range edges {
		fmap, ok := gi.Following.Load(e.From)
		if !ok {
			fmap = roaring64.New()
			gi.Following.Store(e.From, fmap)
		}
		gi.followingLks[e.From%numLocks].Lock()
		fmap.Add(uint64(e.To))
		gi.followingLks[e.From%numLocks].Unlock()
	}

	// Add to inverted index
	for _, e := range edges {
		fmap, ok := gi.Followers.Load(e.To)
		if !ok {
			fmap = roaring64.New()
			gi.Followers.Store(e.To, fmap)
		}
		gi.followersLks[e.To%numLocks].Lock()
		fmap.Add(uint64(e.From))
		gi.followersLks[e.To%numLocks].Unlock()
	}

	gi.mu.Lock()
	defer gi.mu.Unlock()
	// Ensure users exist
	for _, e := range edges {
		if _, exists := gi.Users[e.From]; !exists {
			gi.Users[e.From] = &User{
				ID:            e.From,
				FollowerCount: 0,
				IsInfluencer:  false,
			}
		}

		if touser, exists := gi.Users[e.To]; !exists {
			gi.Users[e.To] = &User{
				ID:            e.To,
				FollowerCount: 1,
				IsInfluencer:  false,
			}
		} else {
			touser.FollowerCount++
		}
	}
}

func (gi *GraphIndex) OptimizeIndex() {
	bar := progressbar.Default(-1)

	sema := make(chan struct{}, 100)
	var wg sync.WaitGroup
	gi.Following.Range(func(user int64, vals *roaring64.Bitmap) bool {
		wg.Add(1)
		sema <- struct{}{}
		go func(bm *roaring64.Bitmap) {
			defer func() {
				bar.Add(1)
				<-sema
				wg.Done()
			}()
			bm.RunOptimize()
		}(vals)
		return true
	})

	gi.Followers.Range(func(user int64, vals *roaring64.Bitmap) bool {
		wg.Add(1)
		sema <- struct{}{}
		go func(bm *roaring64.Bitmap) {
			defer func() {
				bar.Add(1)
				<-sema
				wg.Done()
			}()
			bm.RunOptimize()
		}(vals)
		return true
	})

	wg.Wait()
}

// GetFollowing returns the bitmap of users that a given user follows
func (gi *GraphIndex) GetFollowing(userID int64) *roaring64.Bitmap {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if following, exists := gi.Following.Load(userID); exists {
		return following
	}
	return roaring64.New()
}

// GetFollowers returns the bitmap of followers for a given user
func (gi *GraphIndex) GetFollowers(userID int64) *roaring64.Bitmap {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if followers, exists := gi.Followers.Load(userID); exists {
		return followers
	}
	return roaring64.New()
}

// GetUser returns user metadata
func (gi *GraphIndex) GetUser(userID int64) *User {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	return gi.Users[userID]
}

// SetUser updates user metadata
func (gi *GraphIndex) SetUser(user *User) {
	gi.mu.Lock()
	defer gi.mu.Unlock()

	gi.Users[user.ID] = user
}

// MarkInfluencer marks a user as an influencer
func (gi *GraphIndex) MarkInfluencer(userID int64) {
	gi.mu.Lock()
	defer gi.mu.Unlock()

	gi.Influencers[userID] = true
	if user, exists := gi.Users[userID]; exists {
		user.IsInfluencer = true
	}
}

// GetInfluencers returns all influencer user IDs
func (gi *GraphIndex) GetInfluencers() []int64 {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	result := make([]int64, 0, len(gi.Influencers))
	for uid := range gi.Influencers {
		result = append(result, uid)
	}
	return result
}

// GetFollowingCount returns the number of users a given user follows
func (gi *GraphIndex) GetFollowingCount(userID int64) int {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if following, exists := gi.Following.Load(userID); exists {
		return int(following.GetCardinality())
	}
	return 0
}

// GetFollowerCount returns the number of followers for a given user
func (gi *GraphIndex) GetFollowerCount(userID int64) int32 {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if user, exists := gi.Users[userID]; exists {
		return user.FollowerCount
	}
	return 0
}
