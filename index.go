package simclusters

import (
	"sync"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// GraphIndex maintains efficient indices for the follow graph
type GraphIndex struct {
	mu sync.RWMutex

	// Forward index: follower -> list of users they follow
	Following   map[int64]*roaring64.Bitmap
	followingLk sync.Mutex

	// Inverted index: followed (influencer) -> list of followers
	Followers   map[int64]*roaring64.Bitmap
	followersLk sync.Mutex

	// User metadata
	Users map[int64]*User

	// Influencer tracking
	Influencers map[int64]bool

	// Deduplication mapping: user ID -> canonical user ID
	CanonicalMapping map[int64]int64
}

// NewGraphIndex creates a new graph index
func NewGraphIndex() *GraphIndex {
	return &GraphIndex{
		Following:        make(map[int64]*roaring64.Bitmap),
		Followers:        make(map[int64]*roaring64.Bitmap),
		Users:            make(map[int64]*User),
		Influencers:      make(map[int64]bool),
		CanonicalMapping: make(map[int64]int64),
	}
}

// AddEdge adds a follow edge to the index
func (gi *GraphIndex) AddEdge(from, to int64) {
	// Add to forward index
	gi.followingLk.Lock()
	if _, exists := gi.Following[from]; !exists {
		gi.Following[from] = roaring64.New()
	}
	gi.Following[from].Add(uint64(to))
	gi.followingLk.Unlock()

	// Add to inverted index
	gi.followersLk.Lock()
	if _, exists := gi.Followers[to]; !exists {
		gi.Followers[to] = roaring64.New()
	}
	gi.Followers[to].Add(uint64(from))
	gi.followersLk.Unlock()

	gi.mu.Lock()
	defer gi.mu.Unlock()
	// Ensure users exist
	if _, exists := gi.Users[from]; !exists {
		gi.Users[from] = &User{
			ID:            from,
			FollowerCount: 0,
			IsInfluencer:  false,
			CanonicalID:   0,
		}
	}

	if touser, exists := gi.Users[to]; !exists {
		gi.Users[to] = &User{
			ID:            to,
			FollowerCount: 1,
			IsInfluencer:  false,
			CanonicalID:   0,
		}
	} else {
		touser.FollowerCount++
	}
}

func (gi *GraphIndex) AddEdges(edges []Edge) {
	// Add to forward index
	gi.followingLk.Lock()
	for _, e := range edges {
		if _, exists := gi.Following[e.From]; !exists {
			gi.Following[e.From] = roaring64.New()
		}
		gi.Following[e.From].Add(uint64(e.To))
	}
	gi.followingLk.Unlock()

	// Add to inverted index
	gi.followersLk.Lock()
	for _, e := range edges {
		if _, exists := gi.Followers[e.To]; !exists {
			gi.Followers[e.To] = roaring64.New()
		}
		gi.Followers[e.To].Add(uint64(e.From))
	}
	gi.followersLk.Unlock()

	gi.mu.Lock()
	defer gi.mu.Unlock()
	// Ensure users exist
	for _, e := range edges {
		if _, exists := gi.Users[e.From]; !exists {
			gi.Users[e.From] = &User{
				ID:            e.From,
				FollowerCount: 0,
				IsInfluencer:  false,
				CanonicalID:   0,
			}
		}

		if touser, exists := gi.Users[e.To]; !exists {
			gi.Users[e.To] = &User{
				ID:            e.To,
				FollowerCount: 1,
				IsInfluencer:  false,
				CanonicalID:   0,
			}
		} else {
			touser.FollowerCount++
		}
	}
}

// GetFollowing returns the bitmap of users that a given user follows
func (gi *GraphIndex) GetFollowing(userID int64) *roaring64.Bitmap {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if following, exists := gi.Following[userID]; exists {
		return following
	}
	return roaring64.New()
}

// GetFollowers returns the bitmap of followers for a given user
func (gi *GraphIndex) GetFollowers(userID int64) *roaring64.Bitmap {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if followers, exists := gi.Followers[userID]; exists {
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

// SetCanonicalMapping sets the canonical mapping for a deduplicated user
func (gi *GraphIndex) SetCanonicalMapping(userID, canonicalID int64) {
	gi.mu.Lock()
	defer gi.mu.Unlock()

	gi.CanonicalMapping[userID] = canonicalID
	if user, exists := gi.Users[userID]; exists {
		user.CanonicalID = canonicalID
	}
}

// GetCanonicalID returns the canonical user ID for a given user
func (gi *GraphIndex) GetCanonicalID(userID int64) int64 {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if canonical, exists := gi.CanonicalMapping[userID]; exists {
		return canonical
	}
	return userID // Self is canonical
}

// GetFollowingCount returns the number of users a given user follows
func (gi *GraphIndex) GetFollowingCount(userID int64) int {
	gi.mu.RLock()
	defer gi.mu.RUnlock()

	if following, exists := gi.Following[userID]; exists {
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
