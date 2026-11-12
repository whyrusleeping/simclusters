package simclusters

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

const checkpointVersion = 3
const checkpointBufferSize = 256 * 1024

// CheckpointState represents the state to be saved/loaded.
type CheckpointState struct {
	Influencers []int64
	// sim is the influencer similarity graph: all pairs with similarity of
	// at least SimilarityFloor. Nil means no pairs.
	sim             *simMatrix
	SimilarityFloor float64
	// GraphIndex holds the follow graph.
	GraphIndex *GraphIndex
}

// NumSimilarityEdges returns the number of influencer pairs in the similarity
// graph.
func (s *CheckpointState) NumSimilarityEdges() int {
	if s.sim == nil {
		return 0
	}
	return s.sim.numEdges()
}

// SimilarityEdges yields the similarity graph's edges ordered by (A, B).
func (s *CheckpointState) SimilarityEdges() iter.Seq[SimEdge] {
	if s.sim == nil {
		return func(func(SimEdge) bool) {}
	}
	return s.sim.edges()
}

// CheckpointMetadata stores metadata about the checkpoint.
type CheckpointMetadata struct {
	Version         int     `json:"version"`
	Timestamp       int64   `json:"timestamp"`
	NumInfluencers  int     `json:"num_influencers"`
	NumUsers        int     `json:"num_users"`
	NumMatrixEdges  int     `json:"num_matrix_edges"`
	NumFollowEdges  int     `json:"num_follow_edges"`
	SimilarityFloor float64 `json:"similarity_floor"`
}

// Checkpoint files. Every file is little-endian; a section is a uint64 count
// followed by that many fixed-width records.
//
//	influencers.bin  int64 ids
//	matrix.bin       (A int64, B int64, weight float64), ordered by (A, B), A < B
//	users.bin        int64 ids, ascending; a user's position is its index
//	following.bin    a section of uint32 out-degrees, one per user, then a
//	                 section of uint32 followed-user indices, row after row,
//	                 each row ascending
var checkpointFiles = []string{"metadata.json", "influencers.bin", "matrix.bin", "users.bin", "following.bin"}

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
	gi := state.GraphIndex
	gi.build(runtime.GOMAXPROCS(0))
	metadata := CheckpointMetadata{
		Version: checkpointVersion, Timestamp: time.Now().Unix(),
		NumInfluencers: len(state.Influencers), NumMatrixEdges: state.NumSimilarityEdges(),
		NumUsers: len(gi.ids), NumFollowEdges: len(gi.adj),
		SimilarityFloor: state.SimilarityFloor,
	}
	slog.Info("Saving checkpoint", "path", dirPath, "influencers", metadata.NumInfluencers,
		"edges", metadata.NumMatrixEdges, "users", metadata.NumUsers, "follows", metadata.NumFollowEdges)
	putInt64 := func(b []byte, v int64) { binary.LittleEndian.PutUint64(b, uint64(v)) }
	putUint32 := func(b []byte, v uint32) { binary.LittleEndian.PutUint32(b, v) }
	var g errgroup.Group
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "influencers.bin"), func(w io.Writer) error {
			return writeCheckpointSection(w, metadata.NumInfluencers, slices.Values(state.Influencers), 8, putInt64)
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "matrix.bin"), func(w io.Writer) error {
			return writeCheckpointSection(w, metadata.NumMatrixEdges, state.SimilarityEdges(), 24, func(b []byte, e SimEdge) {
				binary.LittleEndian.PutUint64(b, uint64(e.A))
				binary.LittleEndian.PutUint64(b[8:], uint64(e.B))
				binary.LittleEndian.PutUint64(b[16:], math.Float64bits(e.Weight))
			})
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "users.bin"), func(w io.Writer) error {
			return writeCheckpointSection(w, metadata.NumUsers, slices.Values(gi.ids), 8, putInt64)
		})
	})
	g.Go(func() error {
		return writeCheckpointFile(filepath.Join(dirPath, "following.bin"), func(w io.Writer) error {
			degrees := func(yield func(uint32) bool) {
				for u := range gi.ids {
					if !yield(uint32(gi.off[u+1] - gi.off[u])) {
						return
					}
				}
			}
			if err := writeCheckpointSection(w, metadata.NumUsers, degrees, 4, putUint32); err != nil {
				return err
			}
			return writeCheckpointSection(w, metadata.NumFollowEdges, slices.Values(gi.adj), 4, putUint32)
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
	slog.Info("Checkpoint saved", "path", dirPath, "size_bytes", checkpointSize(dirPath),
		"duration", time.Since(start))
	return nil
}

// LoadCheckpoint loads and validates a committed checkpoint.
func LoadCheckpoint(dirPath string) (*CheckpointState, error) {
	start := time.Now()
	data, err := os.ReadFile(filepath.Join(dirPath, "metadata.json"))
	if err != nil {
		return nil, fmt.Errorf("metadata.json: %w", err)
	}
	var m CheckpointMetadata
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("metadata.json: %w", err)
	}
	if m.Version != checkpointVersion {
		return nil, fmt.Errorf("metadata.json: unsupported checkpoint version %d (want %d); rerun from the CSV input to rebuild it",
			m.Version, checkpointVersion)
	}
	if m.NumInfluencers < 0 || m.NumMatrixEdges < 0 || m.NumUsers < 0 || m.NumFollowEdges < 0 {
		return nil, fmt.Errorf("metadata.json: negative record count")
	}
	if m.NumUsers > math.MaxUint32 {
		return nil, fmt.Errorf("metadata.json: too many users")
	}
	slog.Info("Loading checkpoint", "path", dirPath, "influencers", m.NumInfluencers,
		"edges", m.NumMatrixEdges, "users", m.NumUsers, "follows", m.NumFollowEdges, "size_bytes", checkpointSize(dirPath))
	workers := runtime.GOMAXPROCS(0)
	state := &CheckpointState{SimilarityFloor: m.SimilarityFloor}
	var g errgroup.Group
	g.Go(func() error {
		err := readCheckpointFile(filepath.Join(dirPath, "influencers.bin"), []checkpointSection{{m.NumInfluencers, 8}}, func(r io.Reader) error {
			state.Influencers = make([]int64, m.NumInfluencers)
			return readCheckpointSection(r, m.NumInfluencers, 8, func(i int, b []byte) bool {
				state.Influencers[i] = int64(binary.LittleEndian.Uint64(b))
				return true
			})
		})
		if err != nil {
			return err
		}
		ids := slices.Clone(state.Influencers)
		slices.Sort(ids)
		return readCheckpointFile(filepath.Join(dirPath, "matrix.bin"), []checkpointSection{{m.NumMatrixEdges, 24}}, func(r io.Reader) error {
			var readErr error
			edges := func(yield func(SimEdge) bool) {
				readErr = readCheckpointSection(r, m.NumMatrixEdges, 24, func(_ int, b []byte) bool {
					return yield(SimEdge{A: int64(binary.LittleEndian.Uint64(b)), B: int64(binary.LittleEndian.Uint64(b[8:])),
						Weight: math.Float64frombits(binary.LittleEndian.Uint64(b[16:]))})
				})
			}
			sim, err := newSimMatrix(ids, edges)
			if err != nil {
				return err
			}
			state.sim = sim
			return readErr
		})
	})
	gi := &GraphIndex{}
	g.Go(func() error {
		return readCheckpointFile(filepath.Join(dirPath, "users.bin"), []checkpointSection{{m.NumUsers, 8}}, func(r io.Reader) error {
			gi.ids = make([]int64, m.NumUsers)
			if err := readCheckpointSection(r, m.NumUsers, 8, func(i int, b []byte) bool {
				gi.ids[i] = int64(binary.LittleEndian.Uint64(b))
				return true
			}); err != nil {
				return err
			}
			for i := 1; i < len(gi.ids); i++ {
				if gi.ids[i] <= gi.ids[i-1] {
					return errors.New("user ids are not strictly ascending")
				}
			}
			return nil
		})
	})
	g.Go(func() error {
		return readCheckpointFile(filepath.Join(dirPath, "following.bin"),
			[]checkpointSection{{m.NumUsers, 4}, {m.NumFollowEdges, 4}}, func(r io.Reader) error {
				degrees := make([]uint32, m.NumUsers)
				if err := readCheckpointUint32s(r, degrees); err != nil {
					return err
				}
				gi.off = make([]uint64, m.NumUsers+1)
				for u, d := range degrees {
					gi.off[u+1] = gi.off[u] + uint64(d)
				}
				if gi.off[m.NumUsers] != uint64(m.NumFollowEdges) {
					return fmt.Errorf("out-degrees sum to %d, want %d", gi.off[m.NumUsers], m.NumFollowEdges)
				}
				gi.adj = make([]uint32, m.NumFollowEdges)
				if err := readCheckpointUint32s(r, gi.adj); err != nil {
					return err
				}
				var bad atomic.Int64
				bad.Store(-1)
				parallelRange(m.NumUsers, workers, 4096, func(_, lo, hi int) {
					for u := lo; u < hi; u++ {
						row := gi.row(uint32(u))
						for i, v := range row {
							if int(v) >= m.NumUsers || i > 0 && v <= row[i-1] {
								bad.Store(int64(u))
								return
							}
						}
					}
				})
				if u := bad.Load(); u >= 0 {
					return fmt.Errorf("row of user index %d is out of range or not strictly ascending", u)
				}
				return nil
			})
	})
	if err := g.Wait(); err != nil {
		return nil, err
	}
	gi.inDeg = countInDegrees(len(gi.ids), gi.adj, workers)
	state.GraphIndex = gi
	slog.Info("Checkpoint loaded", "path", dirPath, "influencers", len(state.Influencers),
		"edges", m.NumMatrixEdges, "users", m.NumUsers, "follows", m.NumFollowEdges, "duration", time.Since(start))
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

func checkpointSize(dir string) int64 {
	var size int64
	for _, name := range checkpointFiles {
		if info, err := os.Stat(filepath.Join(dir, name)); err == nil {
			size += info.Size()
		}
	}
	return size
}

// writeCheckpointSection writes a section of count records.
func writeCheckpointSection[T any](w io.Writer, count int, records iter.Seq[T], width int, encode func([]byte, T)) error {
	var header [8]byte
	binary.LittleEndian.PutUint64(header[:], uint64(count))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	buf := make([]byte, (checkpointBufferSize/width)*width)
	pos, written := 0, 0
	for record := range records {
		encode(buf[pos:pos+width], record)
		pos += width
		written++
		if pos == len(buf) {
			if _, err := w.Write(buf); err != nil {
				return err
			}
			pos = 0
		}
	}
	if written != count {
		return fmt.Errorf("wrote %d records, expected %d", written, count)
	}
	_, err := w.Write(buf[:pos])
	return err
}

// checkpointSection describes a section's record count and width.
type checkpointSection struct{ count, width int }

// readCheckpointFile checks that the file at path holds exactly the given
// sections, then passes it to read. Errors are prefixed with the path.
func readCheckpointFile(path string, sections []checkpointSection, read func(io.Reader) error) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s: %w", path, err)
		}
	}()
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, f.Close()) }()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	remaining := info.Size()
	for _, s := range sections {
		// Division rather than multiplication avoids overflow on a corrupt count.
		if remaining -= 8; remaining < 0 || int64(s.count) > remaining/int64(s.width) {
			return fmt.Errorf("file size %d too small for its records", info.Size())
		}
		remaining -= int64(s.count) * int64(s.width)
	}
	r := bufio.NewReaderSize(f, checkpointBufferSize)
	if err := read(r); err != nil {
		return err
	}
	return checkpointEOF(r)
}

// readCheckpointSection reads a section that must hold expected records and
// passes each to decode, stopping early if decode returns false.
func readCheckpointSection(r io.Reader, expected, width int, decode func(i int, b []byte) bool) error {
	var header [8]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return err
	}
	if count := binary.LittleEndian.Uint64(header[:]); count != uint64(expected) {
		return fmt.Errorf("record count %d, expected %d", count, expected)
	}
	buf := make([]byte, (checkpointBufferSize/width)*width)
	for lo := 0; lo < expected; {
		n := min(len(buf)/width, expected-lo)
		if _, err := io.ReadFull(r, buf[:n*width]); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			if !decode(lo+i, buf[i*width:(i+1)*width]) {
				return nil
			}
		}
		lo += n
	}
	return nil
}

// readCheckpointUint32s reads a section of uint32 records into dst, which
// must have the expected length.
func readCheckpointUint32s(r io.Reader, dst []uint32) error {
	var header [8]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return err
	}
	if count := binary.LittleEndian.Uint64(header[:]); count != uint64(len(dst)) {
		return fmt.Errorf("record count %d, expected %d", count, len(dst))
	}
	buf := make([]byte, checkpointBufferSize)
	for len(dst) > 0 {
		n := min(len(buf)/4, len(dst))
		if _, err := io.ReadFull(r, buf[:4*n]); err != nil {
			return err
		}
		for i := range dst[:n] {
			dst[i] = binary.LittleEndian.Uint32(buf[4*i:])
		}
		dst = dst[n:]
	}
	return nil
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
