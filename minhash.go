package simclusters

import (
	"hash/fnv"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// MinHashSignature represents a MinHash sketch for fast similarity estimation
type MinHashSignature struct {
	Hashes  []uint32
	NumHash int
}

// ComputeMinHash computes MinHash signature for a roaring bitmap
func ComputeMinHash(bitmap *roaring64.Bitmap, numHashes int) *MinHashSignature {
	sig := &MinHashSignature{
		Hashes:  make([]uint32, numHashes),
		NumHash: numHashes,
	}

	// Initialize with max values
	for i := range sig.Hashes {
		sig.Hashes[i] = ^uint32(0) // Max uint32
	}

	// Iterate through all elements in the bitmap
	iter := bitmap.Iterator()
	for iter.HasNext() {
		elem := iter.Next()

		// Apply each hash function
		for i := 0; i < numHashes; i++ {
			// Hash the element with seed i
			h := hashWithSeed(elem, uint32(i))

			// Keep minimum hash value for this position
			if h < sig.Hashes[i] {
				sig.Hashes[i] = h
			}
		}
	}

	return sig
}

// hashWithSeed creates a hash of value with given seed
func hashWithSeed(value uint64, seed uint32) uint32 {
	h := fnv.New32a()

	// Write seed
	h.Write([]byte{
		byte(seed >> 24),
		byte(seed >> 16),
		byte(seed >> 8),
		byte(seed),
	})

	// Write value
	h.Write([]byte{
		byte(value >> 24),
		byte(value >> 16),
		byte(value >> 8),
		byte(value),
	})

	return h.Sum32()
}

// EstimateSimilarity estimates Jaccard similarity between two MinHash signatures
func EstimateSimilarity(sig1, sig2 *MinHashSignature) float64 {
	if sig1.NumHash != sig2.NumHash {
		return 0
	}

	matches := countMatchesScalar(sig1.Hashes, sig2.Hashes)

	return float64(matches) / float64(sig1.NumHash)
}

func countMatchesScalar(a, b []uint32) int {
	n := len(a)
	matches := 0

	// Process 8 at a time (unrolled for better ILP)
	i := 0
	for ; i+7 < n; i += 8 {
		// Modern CPUs can execute these comparisons in parallel
		if a[i] == b[i] {
			matches++
		}
		if a[i+1] == b[i+1] {
			matches++
		}
		if a[i+2] == b[i+2] {
			matches++
		}
		if a[i+3] == b[i+3] {
			matches++
		}
		if a[i+4] == b[i+4] {
			matches++
		}
		if a[i+5] == b[i+5] {
			matches++
		}
		if a[i+6] == b[i+6] {
			matches++
		}
		if a[i+7] == b[i+7] {
			matches++
		}
	}

	// Handle remainder
	for ; i < n; i++ {
		if a[i] == b[i] {
			matches++
		}
	}

	return matches
}
