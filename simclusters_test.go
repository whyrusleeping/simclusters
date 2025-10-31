package simclusters

import (
	"strings"
	"testing"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

func TestEnumerateSubsets(t *testing.T) {
	tests := []struct {
		name             string
		communities      []uint64
		maxSize          int
		expectedCount    int
		expectedContents [][]uint64 // For small cases, we can check exact contents
	}{
		{
			name:          "3 communities, max size 2",
			communities:   []uint64{0, 1, 2},
			maxSize:       2,
			expectedCount: 6, // C(3,0) + C(3,1) + C(3,2) = 1 + 3 + 3 = 7
			expectedContents: [][]uint64{
				//{},  // empty set
				{0}, // singles
				{1},
				{2},
				{0, 1}, // pairs
				{0, 2},
				{1, 2},
			},
		},
		{
			name:          "4 communities, max size 2",
			communities:   []uint64{10, 20, 30, 40},
			maxSize:       2,
			expectedCount: 10, // C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11
		},
		{
			name:          "5 communities, max size 3",
			communities:   []uint64{5, 10, 15, 20, 25},
			maxSize:       3,
			expectedCount: 25, // C(5,0) + C(5,1) + C(5,2) + C(5,3) = 1 + 5 + 10 + 10 = 26
		},
		{
			name:          "2 communities, max size 4 (capped at 2)",
			communities:   []uint64{100, 200},
			maxSize:       4,
			expectedCount: 3, // C(2,0) + C(2,1) + C(2,2) = 1 + 2 + 1 = 4
		},
		{
			name:          "single community",
			communities:   []uint64{42},
			maxSize:       1,
			expectedCount: 1, // C(1,0) + C(1,1) = 1 + 1 = 2 (empty and {42})
		},
		{
			name:          "4 communities, max size 4 (all subsets)",
			communities:   []uint64{1, 2, 3, 4},
			maxSize:       4,
			expectedCount: 15, // 2^4 = 16 (all possible subsets)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var results []*roaring64.Bitmap
			seen := make(map[string]bool) // Use string representation to detect duplicates

			// Collect all subsets
			enumerateSubsets(tt.communities, tt.maxSize, func(bm *roaring64.Bitmap) {
				// Clone the bitmap since the callback reuses the same one
				cloned := bm.Clone()
				results = append(results, cloned)

				// Check for duplicates using bitmap's String representation
				key := cloned.String()
				if seen[key] {
					t.Errorf("Duplicate subset generated: %v", bitmapToSlice(cloned))
				}
				seen[key] = true

				// Verify size constraint
				size := cloned.GetCardinality()
				if size > uint64(tt.maxSize) {
					t.Errorf("Subset size %d exceeds maxSize %d: %v",
						size, tt.maxSize, bitmapToSlice(cloned))
				}

				// Verify all elements are from the input communities
				iter := cloned.Iterator()
				for iter.HasNext() {
					val := iter.Next()
					found := false
					for _, comm := range tt.communities {
						if val == comm {
							found = true
							break
						}
					}
					if !found {
						t.Errorf("Subset contains element %d not in input communities", val)
					}
				}
			})

			// Check count
			if len(results) != tt.expectedCount {
				t.Errorf("Expected %d subsets, got %d", tt.expectedCount, len(results))
			}

			// For small test cases, verify exact contents
			if tt.expectedContents != nil {
				if !verifySubsetContents(t, results, tt.expectedContents) {
					t.Errorf("Subset contents don't match expected")
				}
			}

			t.Logf("✓ Generated %d unique subsets (expected %d)",
				len(results), tt.expectedCount)
		})
	}
}

// Helper function to convert bitmap to sorted slice for comparison
func bitmapToSlice(bm *roaring64.Bitmap) []uint64 {
	result := make([]uint64, 0, bm.GetCardinality())
	iter := bm.Iterator()
	for iter.HasNext() {
		result = append(result, iter.Next())
	}
	return result
}

// Helper function to verify that generated subsets match expected contents
func verifySubsetContents(t *testing.T, results []*roaring64.Bitmap, expected [][]uint64) bool {
	if len(results) != len(expected) {
		t.Logf("Count mismatch: got %d, expected %d", len(results), len(expected))
		return false
	}

	// Convert results to comparable format
	resultSets := make(map[string][]uint64)
	for _, bm := range results {
		slice := bitmapToSlice(bm)
		key := bm.String()
		resultSets[key] = slice
	}

	// Convert expected to bitmaps and check
	for _, expectedSlice := range expected {
		bm := roaring64.New()
		for _, val := range expectedSlice {
			bm.Add(val)
		}
		key := bm.String()

		if _, found := resultSets[key]; !found {
			t.Logf("Missing expected subset: %v", expectedSlice)
			return false
		}
	}

	return true
}

// Test that enumerateSubsets produces consistent results
func TestEnumerateSubsetsConsistency(t *testing.T) {
	communities := []uint64{0, 1, 2, 3}
	maxSize := 3

	// Run enumeration twice
	var results1, results2 []string

	enumerateSubsets(communities, maxSize, func(bm *roaring64.Bitmap) {
		results1 = append(results1, bm.Clone().String())
	})

	enumerateSubsets(communities, maxSize, func(bm *roaring64.Bitmap) {
		results2 = append(results2, bm.Clone().String())
	})

	// Results should be identical (same order)
	if len(results1) != len(results2) {
		t.Errorf("Inconsistent number of results: %d vs %d", len(results1), len(results2))
	}

	for i := 0; i < len(results1) && i < len(results2); i++ {
		if results1[i] != results2[i] {
			t.Errorf("Inconsistent result at index %d: %s vs %s", i, results1[i], results2[i])
		}
	}
}

// TestHighFollowerEdgeDistribution analyzes edge cosine similarity for users with high follower counts
func TestHighFollowerEdgeDistribution(t *testing.T) {
	t.Skip()
	checkpointPath := "checkpoint-big5"
	minFollowerThreshold := int32(100000) // Analyze users with 100k+ followers

	t.Logf("Loading checkpoint from %s...", checkpointPath)
	state, err := LoadCheckpoint(checkpointPath)
	if err != nil {
		t.Skipf("Checkpoint not found, skipping: %v", err)
	}

	t.Logf("Loaded checkpoint with %d influencers, %d edges",
		len(state.Influencers), len(state.InfluencerMatrix))

	// Find high-follower users
	highFollowerUsers := make(map[int64]int32)
	for _, userID := range state.Influencers {
		user := state.GraphIndex.Users[userID]
		if user != nil && user.FollowerCount >= minFollowerThreshold {
			highFollowerUsers[userID] = user.FollowerCount
		}
	}

	t.Logf("Found %d users with >= %d followers", len(highFollowerUsers), minFollowerThreshold)

	if len(highFollowerUsers) == 0 {
		t.Skip("No high-follower users found")
	}

	// Analyze edges for these users
	type EdgeStats struct {
		UserID        int64
		FollowerCount int32
		TotalEdges    int
		EdgesUnder01  int
		EdgesUnder015 int
		EdgesUnder02  int
		Edges02to03   int
		EdgesOver03   int
		MinSimilarity float64
		MaxSimilarity float64
		AvgSimilarity float64
		SimilaritySum float64
	}

	userStats := make(map[int64]*EdgeStats)
	for userID, followerCount := range highFollowerUsers {
		userStats[userID] = &EdgeStats{
			UserID:        userID,
			FollowerCount: followerCount,
			MinSimilarity: 1.0,
			MaxSimilarity: 0.0,
		}
	}

	// Scan all edges involving high-follower users
	for edge, similarity := range state.InfluencerMatrix {
		var stats *EdgeStats
		var ok bool

		// Check if either end is a high-follower user
		if stats, ok = userStats[edge.A]; ok {
			// edge.A is a high-follower user
		} else if stats, ok = userStats[edge.B]; ok {
			// edge.B is a high-follower user
		} else {
			continue // Neither end is a high-follower user
		}

		stats.TotalEdges++
		stats.SimilaritySum += similarity

		if similarity < stats.MinSimilarity {
			stats.MinSimilarity = similarity
		}
		if similarity > stats.MaxSimilarity {
			stats.MaxSimilarity = similarity
		}

		// Categorize by similarity threshold
		if similarity < 0.1 {
			stats.EdgesUnder01++
		} else if similarity < 0.15 {
			stats.EdgesUnder015++
		} else if similarity < 0.2 {
			stats.EdgesUnder02++
		} else if similarity < 0.3 {
			stats.Edges02to03++
		} else {
			stats.EdgesOver03++
		}
	}

	// Calculate averages and print statistics
	t.Logf("\n=== High-Follower User Edge Statistics ===")
	t.Logf("%-12s %-12s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s",
		"UserID", "Followers", "TotEdges", "<0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", ">0.30", "Min", "Max", "Avg")
	t.Logf("%s", strings.Repeat("-", 120))

	// Sort by follower count for readability
	sortedUsers := make([]*EdgeStats, 0, len(userStats))
	for _, stats := range userStats {
		if stats.TotalEdges > 0 {
			stats.AvgSimilarity = stats.SimilaritySum / float64(stats.TotalEdges)
			sortedUsers = append(sortedUsers, stats)
		}
	}

	// Sort by follower count descending
	for i := 0; i < len(sortedUsers); i++ {
		for j := i + 1; j < len(sortedUsers); j++ {
			if sortedUsers[i].FollowerCount < sortedUsers[j].FollowerCount {
				sortedUsers[i], sortedUsers[j] = sortedUsers[j], sortedUsers[i]
			}
		}
	}

	// Print stats
	for _, stats := range sortedUsers {
		t.Logf("%-12d %-12d %-10d %-8d %-8d %-8d %-8d %-8d %-8.4f %-8.4f %-8.4f",
			stats.UserID,
			stats.FollowerCount,
			stats.TotalEdges,
			stats.EdgesUnder01,
			stats.EdgesUnder015,
			stats.EdgesUnder02,
			stats.Edges02to03,
			stats.EdgesOver03,
			stats.MinSimilarity,
			stats.MaxSimilarity,
			stats.AvgSimilarity)
	}

	// Aggregate statistics
	t.Logf("\n=== Aggregate Statistics ===")
	var totalEdges, totalUnder01, totalUnder015, totalUnder02, total02to03, totalOver03 int
	var totalUsers, usersWithNoEdges, usersLostAt02Threshold int

	for _, stats := range userStats {
		totalUsers++
		if stats.TotalEdges == 0 {
			usersWithNoEdges++
			continue
		}

		totalEdges += stats.TotalEdges
		totalUnder01 += stats.EdgesUnder01
		totalUnder015 += stats.EdgesUnder015
		totalUnder02 += stats.EdgesUnder02
		total02to03 += stats.Edges02to03
		totalOver03 += stats.EdgesOver03

		// Check if user would be isolated at 0.2 threshold
		edgesAt02OrAbove := stats.Edges02to03 + stats.EdgesOver03
		if edgesAt02OrAbove == 0 {
			usersLostAt02Threshold++
		}
	}

	t.Logf("Total high-follower users: %d", totalUsers)
	t.Logf("Users with no edges at all: %d (%.1f%%)", usersWithNoEdges, 100.0*float64(usersWithNoEdges)/float64(totalUsers))
	t.Logf("Users lost at 0.2 threshold: %d (%.1f%%)", usersLostAt02Threshold, 100.0*float64(usersLostAt02Threshold)/float64(totalUsers))
	t.Logf("\nTotal edges: %d", totalEdges)
	t.Logf("Edges < 0.10: %d (%.1f%%)", totalUnder01, 100.0*float64(totalUnder01)/float64(totalEdges))
	t.Logf("Edges 0.10-0.15: %d (%.1f%%)", totalUnder015, 100.0*float64(totalUnder015)/float64(totalEdges))
	t.Logf("Edges 0.15-0.20: %d (%.1f%%)", totalUnder02, 100.0*float64(totalUnder02)/float64(totalEdges))
	t.Logf("Edges 0.20-0.30: %d (%.1f%%)", total02to03, 100.0*float64(total02to03)/float64(totalEdges))
	t.Logf("Edges > 0.30: %d (%.1f%%)", totalOver03, 100.0*float64(totalOver03)/float64(totalEdges))

	edgesLostAt02 := totalUnder01 + totalUnder015 + totalUnder02
	t.Logf("\nEdges lost at 0.2 threshold: %d (%.1f%%)", edgesLostAt02, 100.0*float64(edgesLostAt02)/float64(totalEdges))
}
