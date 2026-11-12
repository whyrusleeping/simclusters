package simclusters

// Edge is a follow relationship in the social graph.
type Edge struct {
	From int64 // follower
	To   int64 // followed user
}

// SimEdge is an undirected similarity edge between two influencers. A < B.
type SimEdge struct {
	A, B   int64
	Weight float64 // cosine similarity of the two follower sets
}
