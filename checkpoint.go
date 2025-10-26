package simclusters

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	"golang.org/x/sync/errgroup"
)

const checkpointVersion = 1

// Progress logging frequency
const progressFreq = 10000

// CheckpointState represents the state to be saved/loaded
type CheckpointState struct {
	Influencers      []int64
	InfluencerMatrix map[matKey]float64
	GraphIndex       *GraphIndex
}

// CheckpointMetadata stores metadata about the checkpoint
type CheckpointMetadata struct {
	Version        int   `json:"version"`
	Timestamp      int64 `json:"timestamp"`
	NumInfluencers int   `json:"num_influencers"`
	NumUsers       int   `json:"num_users"`
	NumMatrixEdges int   `json:"num_matrix_edges"`
	NumFollowing   int   `json:"num_following"`
	NumFollowers   int   `json:"num_followers"`
}

// matKeyJSON is used for JSON serialization of matKey
type matKeyJSON struct {
	A int64   `json:"a"`
	B int64   `json:"b"`
	V float64 `json:"v"`
}

// SaveCheckpoint saves the state to a directory
func SaveCheckpoint(dirPath string, state *CheckpointState) error {
	slog.Info("Saving checkpoint", "path", dirPath)
	start := time.Now()

	// Create checkpoint directory
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// Count following/followers for metadata
	numFollowing := 0
	state.GraphIndex.Following.Range(func(_ int64, _ *roaring64.Bitmap) bool {
		numFollowing++
		return true
	})
	numFollowers := 0
	state.GraphIndex.Followers.Range(func(_ int64, _ *roaring64.Bitmap) bool {
		numFollowers++
		return true
	})

	var errgrp errgroup.Group

	// Save metadata
	metadata := CheckpointMetadata{
		Version:        checkpointVersion,
		Timestamp:      time.Now().Unix(),
		NumInfluencers: len(state.Influencers),
		NumUsers:       len(state.GraphIndex.Users),
		NumMatrixEdges: len(state.InfluencerMatrix),
		NumFollowing:   numFollowing,
		NumFollowers:   numFollowers,
	}

	errgrp.Go(func() error {
		if err := saveJSON(filepath.Join(dirPath, "metadata.json"), metadata); err != nil {
			return fmt.Errorf("failed to save metadata: %w", err)
		}
		return nil
	})

	// Save influencers (binary format)
	errgrp.Go(func() error {
		slog.Info("Saving influencers", "count", len(state.Influencers))
		if err := saveInfluencers(filepath.Join(dirPath, "influencers.bin"), state.Influencers); err != nil {
			return fmt.Errorf("failed to save influencers: %w", err)
		}
		return nil
	})

	// Save influencer matrix (binary format)
	errgrp.Go(func() error {
		slog.Info("Saving influencer matrix", "edges", len(state.InfluencerMatrix))
		if err := saveMatrix(filepath.Join(dirPath, "matrix.bin"), state.InfluencerMatrix); err != nil {
			return fmt.Errorf("failed to save matrix: %w", err)
		}
		return nil
	})

	// Save users (binary format)
	errgrp.Go(func() error {
		slog.Info("Saving users", "count", len(state.GraphIndex.Users))
		if err := saveUsers(filepath.Join(dirPath, "users.bin"), state.GraphIndex.Users); err != nil {
			return fmt.Errorf("failed to save users: %w", err)
		}
		return nil
	})

	// Save following (binary format)
	errgrp.Go(func() error {
		slog.Info("Saving following relationships", "count", numFollowing)
		if err := saveBitmapMap(filepath.Join(dirPath, "following.bin"), &state.GraphIndex.Following); err != nil {
			return fmt.Errorf("failed to save following: %w", err)
		}
		return nil
	})

	// Save followers (binary format)
	errgrp.Go(func() error {
		slog.Info("Saving follower relationships", "count", numFollowers)
		if err := saveBitmapMap(filepath.Join(dirPath, "followers.bin"), &state.GraphIndex.Followers); err != nil {
			return fmt.Errorf("failed to save followers: %w", err)
		}
		return nil
	})

	if err := errgrp.Wait(); err != nil {
		return err
	}

	// Log checkpoint size
	if dirSize, err := getDirSize(dirPath); err == nil {
		sizeMB := float64(dirSize) / (1024 * 1024)
		slog.Info("Checkpoint saved successfully",
			"path", dirPath,
			"size_mb", fmt.Sprintf("%.2f", sizeMB),
			"took", time.Since(start))
	} else {
		slog.Info("Checkpoint saved successfully", "path", dirPath, "took", time.Since(start))
	}

	return nil
}

// LoadCheckpoint loads the state from a directory
func LoadCheckpoint(dirPath string) (*CheckpointState, error) {
	slog.Info("Loading checkpoint", "path", dirPath)
	start := time.Now()

	// Load and verify metadata
	var metadata CheckpointMetadata
	if err := loadJSON(filepath.Join(dirPath, "metadata.json"), &metadata); err != nil {
		return nil, fmt.Errorf("failed to load metadata: %w", err)
	}
	if metadata.Version != checkpointVersion {
		return nil, fmt.Errorf("checkpoint version mismatch: expected %d, got %d", checkpointVersion, metadata.Version)
	}

	slog.Info("Checkpoint metadata loaded",
		"version", metadata.Version,
		"influencers", metadata.NumInfluencers,
		"users", metadata.NumUsers,
		"matrix_edges", metadata.NumMatrixEdges)

	state := &CheckpointState{
		GraphIndex: NewGraphIndex(),
	}

	var errgrp errgroup.Group

	// Load influencers (binary format)
	errgrp.Go(func() error {
		slog.Info("Loading influencers...")
		influencers, err := loadInfluencersBin(filepath.Join(dirPath, "influencers.bin"))
		if err != nil {
			return fmt.Errorf("failed to load influencers: %w", err)
		}
		state.Influencers = influencers
		slog.Info("Influencers loaded", "count", len(state.Influencers))
		return nil
	})

	// Load influencer matrix (binary format)
	errgrp.Go(func() error {
		slog.Info("Loading influencer matrix...")
		matrix, err := loadMatrixBin(filepath.Join(dirPath, "matrix.bin"))
		if err != nil {
			return fmt.Errorf("failed to load matrix: %w", err)
		}
		state.InfluencerMatrix = matrix
		slog.Info("Influencer matrix loaded", "edges", len(state.InfluencerMatrix))
		return nil
	})

	// Load users (binary format)
	errgrp.Go(func() error {
		slog.Info("Loading users...")
		users, err := loadUsersBin(filepath.Join(dirPath, "users.bin"))
		if err != nil {
			return fmt.Errorf("failed to load users: %w", err)
		}
		state.GraphIndex.Users = users
		slog.Info("Users loaded", "count", len(state.GraphIndex.Users))
		return nil
	})

	// Load following relationships
	errgrp.Go(func() error {
		slog.Info("Loading following relationships...")
		if err := loadBitmapMap(filepath.Join(dirPath, "following.bin"), &state.GraphIndex.Following); err != nil {
			return fmt.Errorf("failed to load following: %w", err)
		}
		return nil
	})

	// Load follower relationships
	errgrp.Go(func() error {
		slog.Info("Loading follower relationships...")
		if err := loadBitmapMap(filepath.Join(dirPath, "followers.bin"), &state.GraphIndex.Followers); err != nil {
			return fmt.Errorf("failed to load followers: %w", err)
		}
		return nil
	})

	if err := errgrp.Wait(); err != nil {
		return nil, err
	}

	// Reconstruct influencers map
	state.GraphIndex.Influencers = make(map[int64]bool, len(state.Influencers))
	for _, infID := range state.Influencers {
		state.GraphIndex.Influencers[infID] = true
	}

	slog.Info("Checkpoint loaded successfully", "path", dirPath, "took", time.Since(start))
	return state, nil
}

// saveJSON saves data as JSON to a file
func saveJSON(path string, data interface{}) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered writer for performance
	w := bufio.NewWriterSize(f, 256*1024)
	defer w.Flush()

	encoder := json.NewEncoder(w)
	if err := encoder.Encode(data); err != nil {
		return err
	}

	return w.Flush()
}

// loadJSON loads JSON data from a file
func loadJSON(path string, data interface{}) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered reader for performance
	r := bufio.NewReaderSize(f, 256*1024)

	decoder := json.NewDecoder(r)
	return decoder.Decode(data)
}

// saveBitmapMap saves a sync map of bitmaps in binary format
// Format: [userID: int64][length: int64][bitmap bytes] repeated
func saveBitmapMap(path string, bitmapMap interface {
	Range(func(int64, *roaring64.Bitmap) bool)
}) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered writer for performance
	w := bufio.NewWriterSize(f, 256*1024)
	defer w.Flush()

	var saveErr error
	count := 0
	bitmapMap.Range(func(userID int64, bitmap *roaring64.Bitmap) bool {
		// Serialize bitmap
		bitmapBytes, err := bitmap.ToBytes()
		if err != nil {
			saveErr = fmt.Errorf("failed to serialize bitmap for user %d: %w", userID, err)
			return false
		}

		// Write userID (int64)
		if err := binary.Write(w, binary.LittleEndian, userID); err != nil {
			saveErr = fmt.Errorf("failed to write userID: %w", err)
			return false
		}

		// Write length (int64)
		length := int64(len(bitmapBytes))
		if err := binary.Write(w, binary.LittleEndian, length); err != nil {
			saveErr = fmt.Errorf("failed to write bitmap length: %w", err)
			return false
		}

		// Write bitmap bytes
		if _, err := w.Write(bitmapBytes); err != nil {
			saveErr = fmt.Errorf("failed to write bitmap data: %w", err)
			return false
		}

		count++
		if count%progressFreq == 0 {
			slog.Debug("Bitmap save progress", "saved", count)
		}
		return true
	})

	if saveErr != nil {
		return saveErr
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("failed to flush bitmap data: %w", err)
	}

	slog.Info("Bitmap map saved", "count", count, "file", filepath.Base(path))
	return nil
}

// loadBitmapMap loads a sync map of bitmaps from binary format
func loadBitmapMap(path string, bitmapMap interface {
	Store(int64, *roaring64.Bitmap)
}) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Use buffered reader for performance
	r := bufio.NewReaderSize(f, 256*1024)

	count := 0
	for {
		// Read userID (int64)
		var userID int64
		if err := binary.Read(r, binary.LittleEndian, &userID); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to read userID: %w", err)
		}

		// Read length (int64)
		var length int64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return fmt.Errorf("failed to read bitmap length: %w", err)
		}

		// Validate length
		if length < 0 || length > 100*1024*1024 { // Max 100MB per bitmap
			return fmt.Errorf("invalid bitmap length %d for user %d", length, userID)
		}

		// Read bitmap bytes - use io.ReadFull to ensure we read exactly length bytes
		bitmapBytes := make([]byte, length)
		if _, err := io.ReadFull(r, bitmapBytes); err != nil {
			return fmt.Errorf("failed to read bitmap data for user %d: %w", userID, err)
		}

		// Deserialize bitmap
		bitmap := roaring64.New()
		if err := bitmap.UnmarshalBinary(bitmapBytes); err != nil {
			return fmt.Errorf("failed to deserialize bitmap for user %d: %w", userID, err)
		}

		bitmapMap.Store(userID, bitmap)

		count++
		if count%progressFreq == 0 {
			slog.Debug("Bitmap load progress", "loaded", count)
		}
	}

	slog.Info("Bitmap map loaded", "count", count, "file", filepath.Base(path))
	return nil
}

// getDirSize calculates the total size of a directory
func getDirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}

// saveInfluencers saves influencer IDs in binary format
// Format: [count: int64][id: int64]...
func saveInfluencers(path string, influencers []int64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 256*1024)
	defer w.Flush()

	// Write count
	count := int64(len(influencers))
	if err := binary.Write(w, binary.LittleEndian, count); err != nil {
		return fmt.Errorf("failed to write count: %w", err)
	}

	// Write influencer IDs
	for i, id := range influencers {
		if err := binary.Write(w, binary.LittleEndian, id); err != nil {
			return fmt.Errorf("failed to write influencer ID: %w", err)
		}
		if (i+1)%progressFreq == 0 {
			slog.Debug("Influencer save progress", "saved", i+1, "total", len(influencers))
		}
	}

	return w.Flush()
}

// loadInfluencersBin loads influencer IDs from binary format
func loadInfluencersBin(path string) ([]int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 256*1024)

	// Read count
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read count: %w", err)
	}

	// Read influencer IDs
	influencers := make([]int64, count)
	for i := int64(0); i < count; i++ {
		if err := binary.Read(r, binary.LittleEndian, &influencers[i]); err != nil {
			return nil, fmt.Errorf("failed to read influencer ID: %w", err)
		}
		if (i+1)%int64(progressFreq) == 0 {
			slog.Debug("Influencer load progress", "loaded", i+1, "total", count)
		}
	}

	return influencers, nil
}

// saveMatrix saves the influencer matrix in binary format
// Format: [count: int64][a: int64][b: int64][val: float64]...
func saveMatrix(path string, matrix map[matKey]float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 256*1024)
	defer w.Flush()

	// Write count
	count := int64(len(matrix))
	if err := binary.Write(w, binary.LittleEndian, count); err != nil {
		return fmt.Errorf("failed to write count: %w", err)
	}

	// Write matrix entries
	i := 0
	for key, val := range matrix {
		if err := binary.Write(w, binary.LittleEndian, key.A); err != nil {
			return fmt.Errorf("failed to write key.A: %w", err)
		}
		if err := binary.Write(w, binary.LittleEndian, key.B); err != nil {
			return fmt.Errorf("failed to write key.B: %w", err)
		}
		if err := binary.Write(w, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write value: %w", err)
		}
		i++
		if i%progressFreq == 0 {
			slog.Debug("Matrix save progress", "saved", i, "total", len(matrix))
		}
	}

	return w.Flush()
}

// loadMatrixBin loads the influencer matrix from binary format
func loadMatrixBin(path string) (map[matKey]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 256*1024)

	// Read count
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read count: %w", err)
	}

	// Read matrix entries
	matrix := make(map[matKey]float64, count)
	for i := int64(0); i < count; i++ {
		var a, b int64
		var val float64

		if err := binary.Read(r, binary.LittleEndian, &a); err != nil {
			return nil, fmt.Errorf("failed to read key.A: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &b); err != nil {
			return nil, fmt.Errorf("failed to read key.B: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &val); err != nil {
			return nil, fmt.Errorf("failed to read value: %w", err)
		}

		matrix[matKey{A: a, B: b}] = val

		if (i+1)%int64(progressFreq) == 0 {
			slog.Debug("Matrix load progress", "loaded", i+1, "total", count)
		}
	}

	return matrix, nil
}

// saveUsers saves user metadata in binary format
// Format: [count: int64][id: int64][followerCount: int32][isInfluencer: byte]...
func saveUsers(path string, users map[int64]*User) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 256*1024)
	defer w.Flush()

	// Write count
	count := int64(len(users))
	if err := binary.Write(w, binary.LittleEndian, count); err != nil {
		return fmt.Errorf("failed to write count: %w", err)
	}

	// Write user entries
	i := 0
	for id, user := range users {
		if err := binary.Write(w, binary.LittleEndian, id); err != nil {
			return fmt.Errorf("failed to write user ID: %w", err)
		}
		if err := binary.Write(w, binary.LittleEndian, user.FollowerCount); err != nil {
			return fmt.Errorf("failed to write follower count: %w", err)
		}
		var isInfluencer byte
		if user.IsInfluencer {
			isInfluencer = 1
		}
		if err := binary.Write(w, binary.LittleEndian, isInfluencer); err != nil {
			return fmt.Errorf("failed to write isInfluencer: %w", err)
		}
		i++
		if i%progressFreq == 0 {
			slog.Debug("User save progress", "saved", i, "total", len(users))
		}
	}

	return w.Flush()
}

// loadUsersBin loads user metadata from binary format
func loadUsersBin(path string) (map[int64]*User, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 256*1024)

	// Read count
	var count int64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("failed to read count: %w", err)
	}

	// Read user entries
	users := make(map[int64]*User, count)
	for i := int64(0); i < count; i++ {
		var id int64
		var followerCount int32
		var isInfluencer byte

		if err := binary.Read(r, binary.LittleEndian, &id); err != nil {
			return nil, fmt.Errorf("failed to read user ID: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &followerCount); err != nil {
			return nil, fmt.Errorf("failed to read follower count: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &isInfluencer); err != nil {
			return nil, fmt.Errorf("failed to read isInfluencer: %w", err)
		}

		users[id] = &User{
			ID:            id,
			FollowerCount: followerCount,
			IsInfluencer:  isInfluencer == 1,
		}

		if (i+1)%int64(progressFreq) == 0 {
			slog.Debug("User load progress", "loaded", i+1, "total", count)
		}
	}

	return users, nil
}
