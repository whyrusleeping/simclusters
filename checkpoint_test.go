package simclusters

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/RoaringBitmap/roaring/v2/roaring64"
	smap "github.com/zolstein/sync-map"
)

func checkpointTestState() *CheckpointState {
	gi := NewGraphIndex()
	gi.AddEdgesBatch([]Edge{{From: 1, To: 2}, {From: 1, To: 3}, {From: 2, To: 3},
		{From: 3, To: 1}, {From: -4, To: 1 << 40}, {From: 1 << 40, To: -4},
		{From: 5, To: 3}, {From: 1, To: 2}})
	return &CheckpointState{
		Influencers: []int64{3, 1 << 40, -4, 2},
		Edges: []SimEdge{{A: 2, B: 3, Weight: 0.75}, {A: -4, B: 1 << 40, Weight: 0.125},
			{A: 2, B: 1 << 40, Weight: math.Copysign(0, -1)}},
		SimilarityFloor: 0.0625, GraphIndex: gi,
	}
}

func saveCheckpointTestState(t *testing.T, dir string, state *CheckpointState) {
	t.Helper()
	if err := SaveCheckpoint(dir, state); err != nil {
		t.Fatal(err)
	}
}

func assertCheckpointMap(t *testing.T, got, want *smap.Map[int64, *roaring64.Bitmap]) {
	t.Helper()
	gotCount, wantCount := 0, 0
	got.Range(func(_ int64, _ *roaring64.Bitmap) bool { gotCount++; return true })
	want.Range(func(id int64, bm *roaring64.Bitmap) bool {
		wantCount++
		actual, ok := got.Load(id)
		if !ok || !actual.Equals(bm) {
			t.Errorf("bitmap mismatch for user %d: got %v, want %v", id, actual, bm)
		}
		return true
	})
	if gotCount != wantCount {
		t.Errorf("map size: got %d, want %d", gotCount, wantCount)
	}
}

func assertCheckpointState(t *testing.T, got, want *CheckpointState, followers bool) {
	t.Helper()
	if !slices.Equal(got.Influencers, want.Influencers) {
		t.Errorf("influencers: got %v, want %v", got.Influencers, want.Influencers)
	}
	if !slices.EqualFunc(got.Edges, want.Edges, func(a, b SimEdge) bool {
		return a.A == b.A && a.B == b.B && math.Float64bits(a.Weight) == math.Float64bits(b.Weight)
	}) {
		t.Errorf("edges differ (including order or weight bits)")
	}
	if got.SimilarityFloor != want.SimilarityFloor {
		t.Errorf("similarity floor: got %g, want %g", got.SimilarityFloor, want.SimilarityFloor)
	}
	assertCheckpointMap(t, &got.GraphIndex.Following, &want.GraphIndex.Following)
	if followers {
		assertCheckpointMap(t, &got.GraphIndex.Followers, &want.GraphIndex.Followers)
	} else {
		got.GraphIndex.Followers.Range(func(id int64, _ *roaring64.Bitmap) bool {
			t.Errorf("unexpected follower entry %d", id)
			return true
		})
	}
}

func readCheckpointTestFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writeCheckpointTestFile(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
}

func expectCheckpointError(t *testing.T, dir, name string) {
	t.Helper()
	state, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err == nil {
		t.Fatalf("expected %s error, got state %v", name, state)
	}
	if !strings.Contains(err.Error(), name) {
		t.Fatalf("error %q does not name %s", err, name)
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	dir := t.TempDir()
	want := checkpointTestState()
	saveCheckpointTestState(t, dir, want)
	got, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)

	// Verify the external-tool formats without using the checkpoint decoder.
	ids := readCheckpointTestFile(t, filepath.Join(dir, "influencers.bin"))
	if int64(binary.LittleEndian.Uint64(ids)) != int64(len(want.Influencers)) {
		t.Fatal("wrong influencer count prefix")
	}
	for i, id := range want.Influencers {
		if int64(binary.LittleEndian.Uint64(ids[8+i*8:])) != id {
			t.Fatalf("influencer %d wire format", i)
		}
	}
	matrix := readCheckpointTestFile(t, filepath.Join(dir, "matrix.bin"))
	if int64(binary.LittleEndian.Uint64(matrix)) != int64(len(want.Edges)) {
		t.Fatal("wrong edge count prefix")
	}
	for i, e := range want.Edges {
		b := matrix[8+i*24:]
		if int64(binary.LittleEndian.Uint64(b)) != e.A || int64(binary.LittleEndian.Uint64(b[8:])) != e.B ||
			binary.LittleEndian.Uint64(b[16:]) != math.Float64bits(e.Weight) {
			t.Fatalf("edge %d wire format", i)
		}
	}
	for name, m := range map[string]*smap.Map[int64, *roaring64.Bitmap]{
		"following.bin": &want.GraphIndex.Following, "followers.bin": &want.GraphIndex.Followers,
	} {
		b := readCheckpointTestFile(t, filepath.Join(dir, name))
		for len(b) > 0 {
			id, length := int64(binary.LittleEndian.Uint64(b)), int(binary.LittleEndian.Uint64(b[8:]))
			bm, ok := m.Load(id)
			if !ok {
				t.Fatalf("%s: unexpected user %d", name, id)
			}
			encoded, err := bm.ToBytes()
			if err != nil || !bytes.Equal(b[16:16+length], encoded) {
				t.Fatalf("%s: incompatible roaring serialization", name)
			}
			b = b[16+length:]
		}
	}
	var metadata CheckpointMetadata
	if err := json.Unmarshal(readCheckpointTestFile(t, filepath.Join(dir, "metadata.json")), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.NumUsers != 6 || metadata.NumInfluencers != len(want.Influencers) || metadata.Version != 2 {
		t.Fatalf("incorrect external-tool metadata: %+v", metadata)
	}
}

func TestCheckpointWithoutFollowers(t *testing.T) {
	dir := t.TempDir()
	want := checkpointTestState()
	saveCheckpointTestState(t, dir, want)
	for _, deleted := range []bool{false, true} {
		if deleted {
			if err := os.Remove(filepath.Join(dir, "followers.bin")); err != nil {
				t.Fatal(err)
			}
		}
		got, err := LoadCheckpoint(dir, LoadOptions{})
		if err != nil {
			t.Fatal(err)
		}
		assertCheckpointState(t, got, want, false)
	}
	expectCheckpointError(t, dir, "followers.bin")
}

func TestCheckpointCorruptData(t *testing.T) {
	for _, name := range []string{"influencers.bin", "matrix.bin", "following.bin", "followers.bin"} {
		for _, corruption := range []string{"mid-record", "record-boundary", "trailing-garbage", "count-mismatch"} {
			t.Run(name+"/"+corruption, func(t *testing.T) {
				dir := t.TempDir()
				saveCheckpointTestState(t, dir, checkpointTestState())
				path := filepath.Join(dir, name)
				b := readCheckpointTestFile(t, path)
				switch corruption {
				case "mid-record":
					b = b[:len(b)-1]
				case "record-boundary":
					switch name {
					case "influencers.bin":
						b = b[:len(b)-8]
					case "matrix.bin":
						b = b[:len(b)-24]
					default:
						last := 0
						for pos := 0; pos < len(b); {
							last = pos
							pos += 16 + int(binary.LittleEndian.Uint64(b[pos+8:]))
						}
						b = b[:last]
					}
				case "trailing-garbage":
					b = append(b, 1, 2, 3)
				case "count-mismatch":
					var m map[string]any
					metaPath := filepath.Join(dir, "metadata.json")
					if err := json.Unmarshal(readCheckpointTestFile(t, metaPath), &m); err != nil {
						t.Fatal(err)
					}
					key := map[string]string{"influencers.bin": "num_influencers", "matrix.bin": "num_matrix_edges",
						"following.bin": "num_following", "followers.bin": "num_followers"}[name]
					m[key] = m[key].(float64) - 1
					encoded, err := json.Marshal(m)
					if err != nil {
						t.Fatal(err)
					}
					writeCheckpointTestFile(t, metaPath, encoded)
				}
				writeCheckpointTestFile(t, path, b)
				expectCheckpointError(t, dir, name)
			})
		}
	}
	for _, name := range []string{"influencers.bin", "matrix.bin"} {
		for _, count := range []uint64{0, math.MaxUint64, math.MaxInt64} {
			t.Run(name+"/bad-count-"+strconv.FormatUint(count, 10), func(t *testing.T) {
				dir := t.TempDir()
				saveCheckpointTestState(t, dir, checkpointTestState())
				path := filepath.Join(dir, name)
				b := readCheckpointTestFile(t, path)
				binary.LittleEndian.PutUint64(b, count)
				writeCheckpointTestFile(t, path, b)
				expectCheckpointError(t, dir, name)
			})
		}
	}
}

func TestCheckpointCorruptBitmaps(t *testing.T) {
	for _, name := range []string{"following.bin", "followers.bin"} {
		for _, corruption := range []string{"negative-length", "oversized-length", "zero-length", "container-count", "bad-bitmap", "internal-trailing", "duplicate-record"} {
			t.Run(name+"/"+corruption, func(t *testing.T) {
				dir := t.TempDir()
				saveCheckpointTestState(t, dir, checkpointTestState())
				path := filepath.Join(dir, name)
				b := readCheckpointTestFile(t, path)
				length := int(binary.LittleEndian.Uint64(b[8:]))
				switch corruption {
				case "negative-length":
					binary.LittleEndian.PutUint64(b[8:], math.MaxUint64)
				case "oversized-length":
					binary.LittleEndian.PutUint64(b[8:], checkpointMaxBitmapSize+1)
				case "zero-length":
					binary.LittleEndian.PutUint64(b[8:], 0)
				case "container-count":
					binary.LittleEndian.PutUint64(b[16:], math.MaxUint64)
				case "bad-bitmap":
					// Corrupt the first embedded 32-bit roaring cookie.
					binary.LittleEndian.PutUint32(b[28:], 0)
				case "internal-trailing":
					pos := 16 + length
					b = append(b[:pos:pos], append([]byte{0x42}, b[pos:]...)...)
					binary.LittleEndian.PutUint64(b[8:], uint64(length+1))
				case "duplicate-record":
					copy(b[16+length:16+length+8], b[:8])
				}
				writeCheckpointTestFile(t, path, b)
				expectCheckpointError(t, dir, name)
			})
		}
	}
}

func TestCheckpointMetadata(t *testing.T) {
	for _, corruption := range []string{"missing", "version", "invalid-json", "trailing-json", "negative-count"} {
		t.Run(corruption, func(t *testing.T) {
			dir := t.TempDir()
			saveCheckpointTestState(t, dir, checkpointTestState())
			path := filepath.Join(dir, "metadata.json")
			b := readCheckpointTestFile(t, path)
			switch corruption {
			case "missing":
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			case "invalid-json":
				writeCheckpointTestFile(t, path, []byte("{"))
			case "trailing-json":
				writeCheckpointTestFile(t, path, append(b, []byte("{}")...))
			default:
				var m map[string]any
				if err := json.Unmarshal(b, &m); err != nil {
					t.Fatal(err)
				}
				if corruption == "version" {
					m["version"] = 99
				} else {
					m["num_following"] = -1
				}
				b, err := json.Marshal(m)
				if err != nil {
					t.Fatal(err)
				}
				writeCheckpointTestFile(t, path, b)
			}
			expectCheckpointError(t, dir, "metadata.json")
		})
	}
}

func TestCheckpointReplaceAndEmpty(t *testing.T) {
	dir := t.TempDir()
	saveCheckpointTestState(t, dir, checkpointTestState())
	gi := NewGraphIndex()
	gi.AddEdgesBatch([]Edge{{From: 100, To: 101}})
	want := &CheckpointState{Influencers: []int64{101}, SimilarityFloor: 0.5, GraphIndex: gi}
	saveCheckpointTestState(t, dir, want)
	got, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)
	want = &CheckpointState{GraphIndex: NewGraphIndex()}
	saveCheckpointTestState(t, dir, want)
	got, err = LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)
}

func TestCheckpointInterruptedSave(t *testing.T) {
	dir := t.TempDir()
	want := checkpointTestState()
	saveCheckpointTestState(t, dir, want)
	if err := os.Remove(filepath.Join(dir, "metadata.json")); err != nil {
		t.Fatal(err)
	}
	writeCheckpointTestFile(t, filepath.Join(dir, ".matrix.bin-stale.tmp"), []byte("partial data"))
	writeCheckpointTestFile(t, filepath.Join(dir, ".metadata.json-stale.tmp"), []byte("{}"))
	expectCheckpointError(t, dir, "metadata.json")
	saveCheckpointTestState(t, dir, want)
	got, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)
}

func TestCheckpointFailedSave(t *testing.T) {
	for _, failure := range []string{"rename", "bitmap-worker", "metadata-encoding"} {
		t.Run(failure, func(t *testing.T) {
			dir := t.TempDir()
			want := checkpointTestState()
			saveCheckpointTestState(t, dir, want)
			switch failure {
			case "rename":
				path := filepath.Join(dir, "matrix.bin")
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
			case "bitmap-worker":
				want.GraphIndex.Following.Store(1234, nil)
			case "metadata-encoding":
				want.SimilarityFloor = math.NaN()
			}
			if err := SaveCheckpoint(dir, want); err == nil {
				t.Fatal("expected save failure")
			}
			expectCheckpointError(t, dir, "metadata.json")
			entries, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range entries {
				if strings.HasSuffix(entry.Name(), ".tmp") {
					t.Errorf("temporary file not cleaned up: %s", entry.Name())
				}
			}
			if failure == "rename" {
				if err := os.Remove(filepath.Join(dir, "matrix.bin")); err != nil {
					t.Fatal(err)
				}
			}
			saveCheckpointTestState(t, dir, checkpointTestState())
			if _, err := LoadCheckpoint(dir, LoadOptions{Followers: true}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCheckpointVersion1(t *testing.T) {
	dir := t.TempDir()
	want := checkpointTestState()
	saveCheckpointTestState(t, dir, want)
	path := filepath.Join(dir, "metadata.json")
	var m map[string]any
	if err := json.Unmarshal(readCheckpointTestFile(t, path), &m); err != nil {
		t.Fatal(err)
	}
	m["version"] = 1
	delete(m, "similarity_floor")
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	writeCheckpointTestFile(t, path, b)
	writeCheckpointTestFile(t, filepath.Join(dir, "users.bin"), []byte("ignored legacy data"))
	want.SimilarityFloor = 0.1 // what version 1 always used
	got, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)
}

func TestCheckpointBulkRecords(t *testing.T) {
	want := &CheckpointState{GraphIndex: NewGraphIndex()}
	for i := range 40000 {
		want.Influencers = append(want.Influencers, int64(i))
		want.Edges = append(want.Edges, SimEdge{A: int64(i), B: int64(i + 1), Weight: float64(i) / 40000})
	}
	var edges []Edge
	for i := range 512 {
		edges = append(edges, Edge{From: int64(i), To: int64(i + 1)})
	}
	want.GraphIndex.AddEdgesBatch(edges)
	dir := t.TempDir()
	saveCheckpointTestState(t, dir, want)
	got, err := LoadCheckpoint(dir, LoadOptions{Followers: true})
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want, true)
}

func TestCheckpointWriterFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "matrix.bin")
	sentinel := errors.New("encode failure")
	err := writeCheckpointFile(path, func(w io.Writer) error {
		if _, err := w.Write([]byte("partial data")); err != nil {
			return err
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), "matrix.bin") {
		t.Fatalf("unexpected error: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("partial/temp file remains: %v, %v", entries, err)
	}
}
