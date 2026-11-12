package simclusters

import (
	"cmp"
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
)

// checkpointTestState has users -4, 1, 2, 3, 5 and 1<<40 (dense indices 0-5)
// with following rows [5] [2 3] [3] [1] [3] [0], and the similarity edges
// (-4, 1<<40), (2, 3), (2, 1<<40) in file order.
func checkpointTestState() *CheckpointState {
	gi := NewGraphIndex()
	gi.AddEdgesBatch([]Edge{{From: 1, To: 2}, {From: 1, To: 3}, {From: 2, To: 3},
		{From: 3, To: 1}, {From: -4, To: 1 << 40}, {From: 1 << 40, To: -4},
		{From: 5, To: 3}, {From: 1, To: 2}})
	influencers := []int64{3, 1 << 40, -4, 2}
	return &CheckpointState{
		Influencers: influencers,
		sim: checkpointTestMatrix(influencers, []SimEdge{{A: 2, B: 3, Weight: 0.75}, {A: -4, B: 1 << 40, Weight: 0.125},
			{A: 2, B: 1 << 40, Weight: math.Copysign(0, -1)}}),
		SimilarityFloor: 0.0625, GraphIndex: gi,
	}
}

func checkpointTestMatrix(influencers []int64, edges []SimEdge) *simMatrix {
	ids := slices.Sorted(slices.Values(influencers))
	edges = slices.Clone(edges)
	slices.SortFunc(edges, func(a, b SimEdge) int { return cmp.Or(cmp.Compare(a.A, b.A), cmp.Compare(a.B, b.B)) })
	m, err := newSimMatrix(ids, slices.Values(edges))
	if err != nil {
		panic(err)
	}
	return m
}

func saveCheckpointTestState(t *testing.T, dir string, state *CheckpointState) {
	t.Helper()
	if err := SaveCheckpoint(dir, state); err != nil {
		t.Fatal(err)
	}
}

func assertCheckpointState(t *testing.T, got, want *CheckpointState) {
	t.Helper()
	if !slices.Equal(got.Influencers, want.Influencers) {
		t.Errorf("influencers: got %v, want %v", got.Influencers, want.Influencers)
	}
	if !slices.EqualFunc(slices.Collect(got.SimilarityEdges()), slices.Collect(want.SimilarityEdges()), func(a, b SimEdge) bool {
		return a.A == b.A && a.B == b.B && math.Float64bits(a.Weight) == math.Float64bits(b.Weight)
	}) {
		t.Errorf("edges differ (including order or weight bits)")
	}
	if got.SimilarityFloor != want.SimilarityFloor {
		t.Errorf("similarity floor: got %g, want %g", got.SimilarityFloor, want.SimilarityFloor)
	}
	g, w := got.GraphIndex, want.GraphIndex
	g.build(1)
	w.build(1)
	if !slices.Equal(g.ids, w.ids) || !slices.Equal(g.adj, w.adj) || !slices.Equal(g.inDeg, w.inDeg) ||
		len(g.ids) > 0 && !slices.Equal(g.off, w.off) {
		t.Errorf("follow graph differs: got ids %v off %v adj %v, want ids %v off %v adj %v", g.ids, g.off, g.adj, w.ids, w.off, w.adj)
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
	state, err := LoadCheckpoint(dir)
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
	got, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want)
	if got.GraphIndex.FollowerCount(3) != 3 || got.GraphIndex.FollowerCount(1<<40) != 1 {
		t.Error("follower counts were not restored")
	}

	// Verify the external-tool formats without using the checkpoint decoder.
	u64 := func(b []byte, i int) uint64 { return binary.LittleEndian.Uint64(b[i:]) }
	u32 := func(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[i:]) }
	ids := readCheckpointTestFile(t, filepath.Join(dir, "influencers.bin"))
	if int(u64(ids, 0)) != len(want.Influencers) {
		t.Fatal("wrong influencer count prefix")
	}
	for i, id := range want.Influencers {
		if int64(u64(ids, 8+i*8)) != id {
			t.Fatalf("influencer %d wire format", i)
		}
	}
	matrix := readCheckpointTestFile(t, filepath.Join(dir, "matrix.bin"))
	edges := slices.Collect(want.SimilarityEdges())
	if int(u64(matrix, 0)) != len(edges) || len(matrix) != 8+24*len(edges) {
		t.Fatal("wrong edge count prefix")
	}
	for i, e := range edges {
		if int64(u64(matrix, 8+i*24)) != e.A || int64(u64(matrix, 16+i*24)) != e.B || u64(matrix, 24+i*24) != math.Float64bits(e.Weight) {
			t.Fatalf("edge %d wire format", i)
		}
	}
	users := readCheckpointTestFile(t, filepath.Join(dir, "users.bin"))
	wantUsers := []int64{-4, 1, 2, 3, 5, 1 << 40}
	if int(u64(users, 0)) != len(wantUsers) || len(users) != 8+8*len(wantUsers) {
		t.Fatal("wrong user count prefix")
	}
	for i, id := range wantUsers {
		if int64(u64(users, 8+i*8)) != id {
			t.Fatalf("user %d wire format", i)
		}
	}
	following := readCheckpointTestFile(t, filepath.Join(dir, "following.bin"))
	degrees, adj := []uint32{1, 2, 1, 1, 1, 1}, []uint32{5, 2, 3, 3, 1, 3, 0}
	if u64(following, 0) != 6 || u64(following, 32) != 7 || len(following) != 40+4*7 {
		t.Fatal("wrong following counts")
	}
	for i, d := range degrees {
		if u32(following, 8+4*i) != d {
			t.Fatalf("degree %d wire format", i)
		}
	}
	for i, v := range adj {
		if u32(following, 40+4*i) != v {
			t.Fatalf("followed user %d wire format", i)
		}
	}
	var metadata CheckpointMetadata
	if err := json.Unmarshal(readCheckpointTestFile(t, filepath.Join(dir, "metadata.json")), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.NumUsers != 6 || metadata.NumFollowEdges != 7 || metadata.NumInfluencers != len(want.Influencers) ||
		metadata.NumMatrixEdges != 3 || metadata.Version != 3 {
		t.Fatalf("incorrect external-tool metadata: %+v", metadata)
	}
}

func TestCheckpointCorruptData(t *testing.T) {
	lastRecord := map[string]int{"influencers.bin": 8, "matrix.bin": 24, "users.bin": 8, "following.bin": 4}
	for _, name := range []string{"influencers.bin", "matrix.bin", "users.bin", "following.bin"} {
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
					b = b[:len(b)-lastRecord[name]]
				case "trailing-garbage":
					b = append(b, 1, 2, 3)
				case "count-mismatch":
					binary.LittleEndian.PutUint64(b, binary.LittleEndian.Uint64(b)-1)
				}
				writeCheckpointTestFile(t, path, b)
				expectCheckpointError(t, dir, name)
			})
		}
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

func TestCheckpointInvalidRecords(t *testing.T) {
	put32 := func(b []byte, i int, v uint32) { binary.LittleEndian.PutUint32(b[i:], v) }
	swap := func(b []byte, i, j, n int) {
		tmp := slices.Clone(b[i : i+n])
		copy(b[i:], b[j:j+n])
		copy(b[j:], tmp)
	}
	for _, tc := range []struct {
		name, file string
		corrupt    func([]byte)
	}{
		{"user-out-of-range", "following.bin", func(b []byte) { put32(b, 40, 6) }},
		{"unsorted-row", "following.bin", func(b []byte) { swap(b, 44, 48, 4) }},
		{"duplicate-in-row", "following.bin", func(b []byte) { put32(b, 48, 2) }},
		{"degree-sum", "following.bin", func(b []byte) { put32(b, 8, 2) }},
		{"users-not-ascending", "users.bin", func(b []byte) { swap(b, 8, 16, 8) }},
		{"not-an-influencer", "matrix.bin", func(b []byte) { binary.LittleEndian.PutUint64(b[8:], 77) }},
		{"edges-out-of-order", "matrix.bin", func(b []byte) { swap(b, 32, 56, 24) }},
		{"reversed-edge", "matrix.bin", func(b []byte) { swap(b, 8, 16, 8) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			saveCheckpointTestState(t, dir, checkpointTestState())
			path := filepath.Join(dir, tc.file)
			b := readCheckpointTestFile(t, path)
			tc.corrupt(b)
			writeCheckpointTestFile(t, path, b)
			expectCheckpointError(t, dir, tc.file)
		})
	}
}

func TestCheckpointMetadata(t *testing.T) {
	for _, corruption := range []string{"missing", "old-version", "unknown-version", "invalid-json", "trailing-json", "negative-count"} {
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
				switch corruption {
				case "old-version":
					m["version"] = 2
				case "unknown-version":
					m["version"] = 99
				default:
					m["num_follow_edges"] = -1
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
	got, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want)
	want = &CheckpointState{GraphIndex: NewGraphIndex()}
	saveCheckpointTestState(t, dir, want)
	got, err = LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want)
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
	got, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want)
}

func TestCheckpointFailedSave(t *testing.T) {
	for _, failure := range []string{"rename", "metadata-encoding"} {
		t.Run(failure, func(t *testing.T) {
			dir := t.TempDir()
			want := checkpointTestState()
			saveCheckpointTestState(t, dir, want)
			switch failure {
			case "rename":
				path := filepath.Join(dir, "following.bin")
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(path, 0700); err != nil {
					t.Fatal(err)
				}
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
				if err := os.Remove(filepath.Join(dir, "following.bin")); err != nil {
					t.Fatal(err)
				}
			}
			saveCheckpointTestState(t, dir, checkpointTestState())
			if _, err := LoadCheckpoint(dir); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCheckpointBulkRecords(t *testing.T) {
	// More records than fit in one I/O buffer.
	want := &CheckpointState{GraphIndex: NewGraphIndex()}
	var simEdges []SimEdge
	for i := range 40000 {
		want.Influencers = append(want.Influencers, int64(i))
		simEdges = append(simEdges, SimEdge{A: int64(i), B: int64(i + 1), Weight: float64(i) / 40000})
	}
	want.Influencers = append(want.Influencers, 40000)
	want.sim = checkpointTestMatrix(want.Influencers, simEdges)
	var edges []Edge
	for i := range 100_000 {
		edges = append(edges, Edge{From: int64(i), To: int64(i+1) % 100_000}, Edge{From: int64(i), To: int64(i*7) % 100_000})
	}
	want.GraphIndex.AddEdgesBatch(edges)
	dir := t.TempDir()
	saveCheckpointTestState(t, dir, want)
	got, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	assertCheckpointState(t, got, want)
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
