package simclusters

// Edge represents a follow relationship in the social graph
type Edge struct {
	From int64 // Follower user ID
	To   int64 // Followed user ID (influencer)
}

// User represents a user in the system with their following pattern
type User struct {
	ID            int64
	FollowerCount int32 // Number of followers (for influencer detection)
	IsInfluencer  bool  // Whether this user is an influencer
	CanonicalID   int64 // If deduplicated, points to canonical user ID (0 if self)
}

// Cluster represents a community/cluster of similar users
type Cluster struct {
	ID          int32             // Cluster ID
	Members     map[int64]float32 // User ID -> membership score
	UpdateEpoch int64             // Last update timestamp for incremental processing
}

// ClusterMembership represents a user's membership in clusters
type ClusterMembership struct {
	UserID   int64
	Clusters map[int32]float32 // Cluster ID -> membership score
}

// Config holds configuration parameters for the SimClusters algorithm
type Config struct {
	// Deduplication settings
	DeduplicationThreshold float32 // Cosine similarity threshold (e.g., 0.85)

	// Influencer detection
	MinFollowerCount int32 // Minimum followers to be considered an influencer

	// Clustering parameters
	NumClusters          int32   // Target number of clusters
	MaxIterations        int32   // Max iterations for clustering algorithm
	ConvergenceThreshold float32 // Convergence threshold for stopping

	// Streaming settings
	BatchSize            int32 // Number of edges to batch before processing
	IncrementalThreshold int32 // Min edges before triggering incremental update

	// Callback settings
	EnableCallbacks bool // Whether to trigger callbacks on changes
}

// DefaultConfig returns a reasonable default configuration
func DefaultConfig() *Config {
	return &Config{
		DeduplicationThreshold: 0.85,
		MinFollowerCount:       100,
		NumClusters:            1000,
		MaxIterations:          20,
		ConvergenceThreshold:   0.001,
		BatchSize:              1000,
		IncrementalThreshold:   100,
	}
}
