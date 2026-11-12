package simclusters

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	smap "github.com/zolstein/sync-map"
	"golang.org/x/sync/errgroup"
)

const checkpointVersion = 2
const checkpointBufferSize = 256 * 1024
const checkpointMaxBitmapSize = 100 * 1024 * 1024

// CheckpointState represents the state to be saved/loaded.
type CheckpointState struct {
	Influencers []int64
	// Edges is the influencer similarity graph: all pairs with similarity
	// of at least SimilarityFloor.
	Edges           []SimEdge
	SimilarityFloor float64
	// GraphIndex holds the follow graph. Followers is only populated on
	// load when LoadOptions.Followers is set.
	GraphIndex *GraphIndex
}

// LoadOptions selects optional parts of a checkpoint to load.
type LoadOptions struct {
	// Followers loads the follower direction, which resuming does not need.
	Followers bool
}

// CheckpointMetadata stores metadata about the checkpoint.
type CheckpointMetadata struct {
	Version         int     `json:"version"`
	Timestamp       int64   `json:"timestamp"`
	NumInfluencers  int     `json:"num_influencers"`
	NumUsers        int     `json:"num_users"`
	NumMatrixEdges  int     `json:"num_matrix_edges"`
	NumFollowing    int     `json:"num_following"`
	NumFollowers    int     `json:"num_followers"`
	SimilarityFloor float64 `json:"similarity_floor"`
}

// SaveCheckpoint saves a quiescent graph. Saves to the same directory must not
// overlap. metadata.json is the commit marker: invalidate it durably before
// replacing any data, and publish it only after all data renames are durable.
func SaveCheckpoint(dirPath string, state *CheckpointState) error {
	start := time.Now()
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return err
	}
	marker := filepath.Join(dirPath, "metadata.json")
	if err := os.Remove(marker); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("metadata.json: %w", err)
	}
	if err := syncCheckpointDir(dirPath); err != nil {
		return fmt.Errorf("metadata.json: invalidating commit: %w", err)
	}
	if state == nil || state.GraphIndex == nil {
		return fmt.Errorf("checkpoint: nil state or GraphIndex")
	}
	metadata := CheckpointMetadata{
		Version: checkpointVersion, Timestamp: time.Now().Unix(),
		NumInfluencers: len(state.Influencers), NumMatrixEdges: len(state.Edges),
		SimilarityFloor: state.SimilarityFloor,
	}
	state.GraphIndex.Following.Range(func(id int64, _ *roaring64.Bitmap) bool {
		metadata.NumFollowing++
		if _, ok := state.GraphIndex.Followers.Load(id); !ok {
			metadata.NumUsers++
		}
		return true
	})
	state.GraphIndex.Followers.Range(func(_ int64, _ *roaring64.Bitmap) bool {
		metadata.NumFollowers++
		metadata.NumUsers++
		return true
	})
	slog.Info("Saving checkpoint", "path", dirPath, "influencers", metadata.NumInfluencers,
		"edges", metadata.NumMatrixEdges, "following", metadata.NumFollowing, "followers", metadata.NumFollowers)
	var g errgroup.Group
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "influencers.bin"), func(w io.Writer) error {
			return writeCheckpointRecords(w, state.Influencers, 8, func(b []byte, id int64) {
				binary.LittleEndian.PutUint64(b, uint64(id))
			})
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "matrix.bin"), func(w io.Writer) error {
			return writeCheckpointRecords(w, state.Edges, 24, func(b []byte, e SimEdge) {
				binary.LittleEndian.PutUint64(b, uint64(e.A))
				binary.LittleEndian.PutUint64(b[8:], uint64(e.B))
				binary.LittleEndian.PutUint64(b[16:], math.Float64bits(e.Weight))
			})
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "following.bin"), func(w io.Writer) error {
			return writeCheckpointBitmaps(w, &state.GraphIndex.Following, metadata.NumFollowing)
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "followers.bin"), func(w io.Writer) error {
			return writeCheckpointBitmaps(w, &state.GraphIndex.Followers, metadata.NumFollowers)
		})
	})
	if err := g.Wait(); err != nil {
		return err
	}
	if err := syncCheckpointDir(dirPath); err != nil {
		return fmt.Errorf("checkpoint data: %w", err)
	}
	if err := writeCheckpointFile(marker, func(w io.Writer) error {
		return json.NewEncoder(w).Encode(metadata)
	}); err != nil {
		return err
	}
	if err := syncCheckpointDir(dirPath); err != nil {
		// Even a failed commit must not leave a loadable marker behind.
		removeErr := os.Remove(marker)
		syncErr := syncCheckpointDir(dirPath)
		return fmt.Errorf("metadata.json: committing: %w", errors.Join(err, removeErr, syncErr))
	}
	slog.Info("Checkpoint saved", "path", dirPath, "influencers", metadata.NumInfluencers,
		"edges", metadata.NumMatrixEdges, "following", metadata.NumFollowing, "followers", metadata.NumFollowers,
		"size_bytes", checkpointSize(dirPath, true), "duration", time.Since(start))
	return nil
}

// LoadCheckpoint loads and validates a committed checkpoint. Version 1's extra
// users.bin is deliberately ignored. Followers is never opened unless requested.
func LoadCheckpoint(dirPath string, opts LoadOptions) (*CheckpointState, error) {
	start := time.Now()
	data, err := os.ReadFile(filepath.Join(dirPath, "metadata.json"))
	if err != nil {
		return nil, fmt.Errorf("metadata.json: %w", err)
	}
	var m CheckpointMetadata
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("metadata.json: %w", err)
	}
	if m.Version != 1 && m.Version != checkpointVersion {
		return nil, fmt.Errorf("metadata.json: unsupported checkpoint version %d", m.Version)
	}
	if m.NumInfluencers < 0 || m.NumMatrixEdges < 0 || m.NumFollowing < 0 || m.NumFollowers < 0 || m.NumUsers < 0 {
		return nil, fmt.Errorf("metadata.json: negative record count")
	}
	if m.Version == 1 {
		// Version 1 did not record the floor; it was always 0.1.
		m.SimilarityFloor = 0.1
	}
	size := checkpointSize(dirPath, opts.Followers)
	slog.Info("Loading checkpoint", "path", dirPath, "influencers", m.NumInfluencers,
		"edges", m.NumMatrixEdges, "following", m.NumFollowing, "followers", m.NumFollowers, "size_bytes", size)
	state := &CheckpointState{GraphIndex: NewGraphIndex(), SimilarityFloor: m.SimilarityFloor}
	var g errgroup.Group
	g.Go(func() error {
		var err error
		state.Influencers, err = readCheckpointRecords(filepath.Join(dirPath, "influencers.bin"), m.NumInfluencers, 8, func(b []byte) int64 {
			return int64(binary.LittleEndian.Uint64(b))
		})
		return err
	})
	g.Go(func() error {
		var err error
		state.Edges, err = readCheckpointRecords(filepath.Join(dirPath, "matrix.bin"), m.NumMatrixEdges, 24, func(b []byte) SimEdge {
			return SimEdge{A: int64(binary.LittleEndian.Uint64(b)), B: int64(binary.LittleEndian.Uint64(b[8:])),
				Weight: math.Float64frombits(binary.LittleEndian.Uint64(b[16:]))}
		})
		return err
	})
	g.Go(func() error {
		return readCheckpointBitmaps(filepath.Join(dirPath, "following.bin"), &state.GraphIndex.Following, m.NumFollowing)
	})
	if opts.Followers {
		g.Go(func() error {
			return readCheckpointBitmaps(filepath.Join(dirPath, "followers.bin"), &state.GraphIndex.Followers, m.NumFollowers)
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	slog.Info("Checkpoint loaded", "path", dirPath, "influencers", len(state.Influencers),
		"edges", len(state.Edges), "following", m.NumFollowing, "followers_loaded", opts.Followers,
		"size_bytes", size, "duration", time.Since(start))
	return state, nil
}

// writeCheckpointFile owns its temporary file until rename. Every failure before
// rename removes it; the caller publishes metadata only after all writers finish.
func writeCheckpointFile(path string, encode func(io.Writer) error) (err error) {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+"-*.tmp")
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	fName := f.Name()
	defer func() {
		if f != nil {
			err = errors.Join(err, f.Close())
		}
		if err != nil {
			if removeErr := os.Remove(fName); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
				err = errors.Join(err, removeErr)
			}
			err = fmt.Errorf("%s: %w", path, err)
		}
	}()
	if err = f.Chmod(0o644); err != nil { // CreateTemp makes it private
		return err
	}
	w := bufio.NewWriterSize(f, checkpointBufferSize)
	if err = encode(w); err != nil {
		return err
	}
	if err = w.Flush(); err != nil {
		return err
	}
	if err = f.Sync(); err != nil {
		return err
	}
	err = f.Close()
	f = nil
	if err != nil {
		return err
	}
	return os.Rename(fName, path)
}

func syncCheckpointDir(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	return errors.Join(f.Sync(), f.Close())
}

func checkpointSize(dir string, followers bool) int64 {
	var size int64
	for _, name := range []string{"metadata.json", "influencers.bin", "matrix.bin", "following.bin", "followers.bin"} {
		if name == "followers.bin" && !followers {
			continue
		}
		if info, err := os.Stat(filepath.Join(dir, name)); err == nil {
			size += info.Size()
		}
	}
	return size
}

func writeCheckpointRecords[T any](w io.Writer, records []T, width int, encode func([]byte, T)) error {
	var header [8]byte
	binary.LittleEndian.PutUint64(header[:], uint64(len(records)))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	buf := make([]byte, (checkpointBufferSize/width)*width)
	for lo := 0; lo < len(records); {
		n := min(len(buf)/width, len(records)-lo)
		for i, record := range records[lo : lo+n] {
			encode(buf[i*width:(i+1)*width], record)
		}
		if _, err := w.Write(buf[:n*width]); err != nil {
			return err
		}
		lo += n
	}
	return nil
}

func readCheckpointRecords[T any](path string, expected, width int, decode func([]byte) T) (records []T, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	defer func() {
		err = errors.Join(err, f.Close())
		if err != nil {
			err = fmt.Errorf("%s: %w", path, err)
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	var header [8]byte
	if _, err := io.ReadFull(f, header[:]); err != nil {
		return nil, err
	}
	count := int64(binary.LittleEndian.Uint64(header[:]))
	// Division rather than multiplication avoids overflow on a corrupt count.
	if count < 0 || count != int64(expected) || info.Size() < 8 ||
		(info.Size()-8)%int64(width) != 0 || (info.Size()-8)/int64(width) != count {
		return nil, fmt.Errorf("invalid count/size: header %d, metadata %d, size %d", count, expected, info.Size())
	}
	records = make([]T, expected)
	buf := make([]byte, (checkpointBufferSize/width)*width)
	for lo := 0; lo < len(records); {
		n := min(len(buf)/width, len(records)-lo)
		if _, err := io.ReadFull(f, buf[:n*width]); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			records[lo+i] = decode(buf[i*width : (i+1)*width])
		}
		lo += n
	}
	return records, checkpointEOF(f)
}

func checkpointEOF(r io.Reader) error {
	var b [1]byte
	n, err := r.Read(b[:])
	if n == 0 && err == io.EOF {
		return nil
	}
	if err != nil {
		return err
	}
	return fmt.Errorf("trailing data after expected records")
}

type checkpointBitmap struct {
	id   int64
	bm   *roaring64.Bitmap
	data []byte
}

func checkpointWorkers() int { return min(runtime.GOMAXPROCS(0), 8) }

// Unbuffered channels bound in-flight bitmap memory to the worker count, not
// the number of graph users. Only the writer goroutine performs sequential I/O.
func writeCheckpointBitmaps(w io.Writer, m *smap.Map[int64, *roaring64.Bitmap], expected int) error {
	g, ctx := errgroup.WithContext(context.Background())
	jobs := make(chan checkpointBitmap)
	results := make(chan checkpointBitmap)
	g.Go(func() error {
		defer close(jobs)
		m.Range(func(id int64, bm *roaring64.Bitmap) bool {
			select {
			case jobs <- checkpointBitmap{id: id, bm: bm}:
				return true
			case <-ctx.Done():
				return false
			}
		})
		return nil
	})
	var workers sync.WaitGroup
	for range checkpointWorkers() {
		workers.Add(1)
		g.Go(func() error {
			defer workers.Done()
			for job := range jobs {
				if job.bm == nil || job.bm.GetSerializedSizeInBytes() > checkpointMaxBitmapSize {
					return fmt.Errorf("user %d: nil or oversized bitmap", job.id)
				}
				data, err := job.bm.ToBytes()
				if err != nil {
					return fmt.Errorf("user %d: %w", job.id, err)
				}
				select {
				case results <- checkpointBitmap{id: job.id, data: data}:
				case <-ctx.Done():
					return ctx.Err()
				}
			}
			return nil
		})
	}
	g.Go(func() error { workers.Wait(); close(results); return nil })
	g.Go(func() error {
		var header [16]byte
		count := 0
		for result := range results {
			binary.LittleEndian.PutUint64(header[:8], uint64(result.id))
			binary.LittleEndian.PutUint64(header[8:], uint64(len(result.data)))
			if _, err := w.Write(header[:]); err != nil {
				return err
			}
			if _, err := w.Write(result.data); err != nil {
				return err
			}
			count++
		}
		if count != expected {
			return fmt.Errorf("record count %d, expected %d", count, expected)
		}
		return nil
	})
	return g.Wait()
}

func readCheckpointBitmaps(path string, m *smap.Map[int64, *roaring64.Bitmap], expected int) (err error) {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	defer func() {
		err = errors.Join(err, f.Close())
		if err != nil {
			err = fmt.Errorf("%s: %w", path, err)
		}
	}()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	// Each record has a 16-byte header and at least an 8-byte empty bitmap.
	if int64(expected) > info.Size()/24 {
		return fmt.Errorf("file too short for %d records", expected)
	}
	g, ctx := errgroup.WithContext(context.Background())
	jobs := make(chan checkpointBitmap)
	g.Go(func() error {
		defer close(jobs)
		r := bufio.NewReaderSize(f, checkpointBufferSize)
		remaining := info.Size()
		var header [16]byte
		for i := 0; i < expected; i++ {
			if _, err := io.ReadFull(r, header[:]); err != nil {
				return err
			}
			remaining -= 16
			id := int64(binary.LittleEndian.Uint64(header[:8]))
			length := int64(binary.LittleEndian.Uint64(header[8:]))
			if length < 8 || length > checkpointMaxBitmapSize || length > remaining {
				return fmt.Errorf("user %d: invalid bitmap length %d (remaining %d)", id, length, remaining)
			}
			data := make([]byte, int(length))
			if _, err := io.ReadFull(r, data); err != nil {
				return err
			}
			remaining -= length
			select {
			case jobs <- checkpointBitmap{id: id, data: data}:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return checkpointEOF(r)
	})
	for range checkpointWorkers() {
		g.Go(func() error {
			for job := range jobs {
				// Roaring64 allocates its outer containers from this count. Bound
				// it by the bytes available before calling into the decoder.
				if binary.LittleEndian.Uint64(job.data[:8]) > uint64((len(job.data)-8)/12) {
					return fmt.Errorf("user %d: invalid bitmap container count", job.id)
				}
				bm := roaring64.New()
				r := bytes.NewReader(job.data)
				if _, err := bm.ReadFrom(r); err != nil {
					return fmt.Errorf("user %d: decoding bitmap: %w", job.id, err)
				}
				if r.Len() != 0 {
					return fmt.Errorf("user %d: trailing bitmap data", job.id)
				}
				if err := bm.Validate(); err != nil {
					return fmt.Errorf("user %d: invalid bitmap: %w", job.id, err)
				}
				if _, loaded := m.LoadOrStore(job.id, bm); loaded {
					return fmt.Errorf("duplicate user %d", job.id)
				}
			}
			return nil
		})
	}
	return g.Wait()
}
