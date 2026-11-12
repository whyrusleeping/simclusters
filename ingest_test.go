package simclusters

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// Deliberately simple and allocation-heavy, independent of the production parser.
func ingestReference(t *testing.T, files []string) []Edge {
	t.Helper()
	var edges []Edge
	for _, contents := range files {
		for lineNo, line := range strings.Split(contents, "\n") {
			line = strings.TrimSuffix(line, "\r")
			if line == "" {
				continue
			}
			fields := strings.Split(line, ",")
			if len(fields) < 3 {
				t.Fatalf("invalid reference row %q", line)
			}
			from, fromErr := strconv.ParseInt(fields[1], 10, 64)
			to, toErr := strconv.ParseInt(fields[2], 10, 64)
			if lineNo == 0 && (fromErr != nil || toErr != nil) {
				continue
			}
			if _, err := strconv.ParseInt(fields[0], 10, 64); err != nil || fromErr != nil || toErr != nil {
				t.Fatalf("invalid reference row %q", line)
			}
			edges = append(edges, Edge{From: from, To: to})
		}
	}
	return edges
}

func ingestCheckGraph(t *testing.T, gi *GraphIndex, edges []Edge) {
	t.Helper()
	following := make(map[int64]map[uint64]bool)
	followers := make(map[int64]map[uint64]bool)
	users := map[int64]bool{0: true, 999999: true}
	for _, e := range edges {
		users[e.From], users[e.To] = true, true
		if following[e.From] == nil {
			following[e.From] = make(map[uint64]bool)
		}
		if followers[e.To] == nil {
			followers[e.To] = make(map[uint64]bool)
		}
		following[e.From][uint64(e.To)] = true
		followers[e.To][uint64(e.From)] = true
	}
	for user := range users {
		for _, direction := range []string{"following", "followers"} {
			wantSet := following[user]
			got := gi.GetFollowing(user)
			if direction == "followers" {
				wantSet = followers[user]
				got = gi.GetFollowers(user)
			}
			want := make([]uint64, 0, len(wantSet))
			for id := range wantSet {
				want = append(want, id)
			}
			slices.Sort(want)
			if got.GetCardinality() != uint64(len(want)) || !slices.Equal(got.ToArray(), want) {
				t.Fatalf("user %d %s = %v, want %v", user, direction, got.ToArray(), want)
			}
		}
	}
}

func ingestWriteFiles(t *testing.T, contents []string) []string {
	t.Helper()
	dir := t.TempDir()
	paths := make([]string, len(contents))
	for i, data := range contents {
		paths[i] = filepath.Join(dir, fmt.Sprintf("edges-%d.csv", i))
		if err := os.WriteFile(paths[i], []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return paths
}

func TestLoadEdgesCSV(t *testing.T) {
	cases := []struct {
		name  string
		files []string
	}{
		{"header", []string{"id,author,subject\n1,10,20\n2,30,20\n"}},
		{"no_header", []string{"1,10,20\n2,20,30\n"}},
		{"CRLF", []string{"id,author,subject\r\n1,10,20\r\n\r\n2,20,30\r\n"}},
		{"blank_lines", []string{"\n\n1,10,20\n\n2,20,30\n\n"}},
		{"no_trailing_newline", []string{"1,10,20\n2,20,30"}},
		{"no_trailing_newline_CR", []string{"1,10,20\r"}},
		{"multiple_files", []string{"id,author,subject\n1,10,20\n", "2,20,30", "", "id,author,subject\r\n3,40,10\r\n"}},
		{"empty", []string{"", "\n\r\n"}},
		{"header_only", []string{"id,author,subject"}},
		{"duplicates", []string{"1,10,20\n1,10,20\n2,10,20\n3,20,10\n"}},
		{"extra_fields", []string{"1,10,20,ignored,negative -1,\n2,30,10,\n"}},
		{"limits_and_leading_zeros", []string{"0,0,9223372036854775807\n00001,00020,00000\n9223372036854775807,20,20\n"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			paths := ingestWriteFiles(t, tc.files)
			want := ingestReference(t, tc.files)
			gi := NewGraphIndex()
			var progress atomic.Int64
			count, err := LoadEdgesCSV(gi, paths, IngestOptions{
				Workers: 4, ChunkBytes: 7, BatchEdges: 2,
				Progress: func(n int64) {
					if n <= 0 {
						t.Errorf("nonpositive progress delta: %d", n)
					}
					progress.Add(n)
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			if count != int64(len(want)) {
				t.Fatalf("count = %d, want %d", count, len(want))
			}
			var total int64
			for _, file := range tc.files {
				total += int64(len(file))
			}
			if progress.Load() != total {
				t.Fatalf("progress = %d, want %d", progress.Load(), total)
			}
			ingestCheckGraph(t, gi, want)
		})
	}
}

func TestIngestChunkBoundaries(t *testing.T) {
	// Includes consecutive newlines, CRLF, a line longer than small chunks, a
	// duplicate, and an unterminated final line. Every possible boundary is tried.
	contents := "id,author,subject\r\n\n1,12345,8\n2,8,9\r\n3,12345,8\n4,0,8"
	paths := ingestWriteFiles(t, []string{contents})
	want := ingestReference(t, []string{contents})
	for chunk := 1; chunk <= len(contents)+1; chunk++ {
		for _, workers := range []int{1, 2, 7} {
			t.Run(fmt.Sprintf("chunk_%d/workers_%d", chunk, workers), func(t *testing.T) {
				gi := NewGraphIndex()
				var progress atomic.Int64
				count, err := LoadEdgesCSV(gi, paths, IngestOptions{
					Workers: workers, ChunkBytes: int64(chunk), BatchEdges: 3,
					Progress: func(n int64) { progress.Add(n) },
				})
				if err != nil || count != int64(len(want)) {
					t.Fatalf("count=%d error=%v, want count=%d", count, err, len(want))
				}
				if progress.Load() != int64(len(contents)) {
					t.Fatalf("progress = %d, want %d", progress.Load(), len(contents))
				}
				ingestCheckGraph(t, gi, want)
			})
		}
	}
}

func TestIngestLongLines(t *testing.T) {
	// Cross the fixed read buffer in numeric fields, ignored fields, and at a
	// CRLF boundary. Leading zeroes remain legal without consuming extra memory.
	const bufferSize = 256 << 10
	cases := []string{
		"1," + strings.Repeat("0", bufferSize*2) + "10,20\r\n2,20,30",
		"1,10,20," + strings.Repeat("x", bufferSize*2) + "\n2,20,30\n",
		"1," + strings.Repeat("0", bufferSize-8) + "10,20\r\n2,20,30\n",
		"1," + strings.Repeat("0", bufferSize-8) + "10,20\r",
		strings.Repeat("0", bufferSize*2) + "1,10,20\n",
		"1,10," + strings.Repeat("0", bufferSize*2) + "20\n",
	}
	for i, contents := range cases {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			paths := ingestWriteFiles(t, []string{contents})
			gi := NewGraphIndex()
			var progress atomic.Int64
			count, err := LoadEdgesCSV(gi, paths, IngestOptions{
				Workers: 3, ChunkBytes: 17, BatchEdges: 1,
				Progress: func(n int64) { progress.Add(n) },
			})
			want := ingestReference(t, []string{contents})
			if err != nil || count != int64(len(want)) {
				t.Fatalf("count=%d error=%v, want %d", count, err, len(want))
			}
			if progress.Load() != int64(len(contents)) {
				t.Fatalf("progress=%d, want %d", progress.Load(), len(contents))
			}
			ingestCheckGraph(t, gi, want)
		})
	}
}

func TestIngestMalformed(t *testing.T) {
	cases := []struct {
		name string
		row  string
	}{
		{"too_few", "2,30"},
		{"letters_author", "2,abc,30"},
		{"letters_subject", "2,30,abc"},
		{"letters_id", "abc,20,30"},
		{"negative_author", "2,-20,30"},
		{"negative_subject", "2,20,-30"},
		{"negative_id", "-2,20,30"},
		{"plus_sign", "2,+20,30"},
		{"overflow_author", "2,9223372036854775808,30"},
		{"overflow_subject", "2,20,18446744073709551616"},
		{"overflow_id", "9223372036854775808,20,30"},
		{"empty_author", "2,,30"},
		{"empty_subject", "2,20,"},
		{"empty_id", ",20,30"},
		{"embedded_CR", "2,2\r0,30"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, prefix := range []string{"", "1,10,20\n"} {
				// Letters in the author/subject of the very first line designate
				// a header, so test those only after a valid data row.
				if prefix == "" && (strings.HasPrefix(tc.name, "letters_") && tc.name != "letters_id" || tc.name == "embedded_CR") {
					continue
				}
				paths := ingestWriteFiles(t, []string{prefix + tc.row + "\n3,40,50\n"})
				_, err := LoadEdgesCSV(NewGraphIndex(), paths, IngestOptions{Workers: 4, ChunkBytes: 1, BatchEdges: 2})
				if err == nil || !strings.Contains(err.Error(), paths[0]) || !strings.Contains(err.Error(), fmt.Sprintf("byte offset %d:", len(prefix))) {
					t.Fatalf("error = %v, want file %q and offset %d", err, paths[0], len(prefix))
				}
			}
		})
	}
	for _, contents := range []string{"\nid,author,subject\n", "id,author\n"} {
		paths := ingestWriteFiles(t, []string{contents})
		if _, err := LoadEdgesCSV(NewGraphIndex(), paths, IngestOptions{Workers: 2, ChunkBytes: 1}); err == nil || !strings.Contains(err.Error(), paths[0]) {
			t.Fatalf("error = %v, want error naming %q", err, paths[0])
		}
	}
}

func TestIngestMissingAndInvalidOptions(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.csv")
	if _, err := LoadEdgesCSV(NewGraphIndex(), []string{missing}, IngestOptions{Workers: 2}); err == nil || !strings.Contains(err.Error(), missing) || !strings.Contains(err.Error(), "byte offset 0:") {
		t.Fatalf("missing file error = %v", err)
	}
	if _, err := LoadEdgesCSV(NewGraphIndex(), nil, IngestOptions{}); err == nil {
		t.Fatal("empty paths accepted")
	}
	paths := ingestWriteFiles(t, []string{"1,2,3\n"})
	for _, opts := range []IngestOptions{{Workers: -1}, {ChunkBytes: -1}, {BatchEdges: -1}} {
		if _, err := LoadEdgesCSV(NewGraphIndex(), paths, opts); err == nil {
			t.Fatalf("negative options accepted: %+v", opts)
		}
	}
	if _, err := LoadEdgesCSV(nil, paths, IngestOptions{}); err == nil {
		t.Fatal("nil graph accepted")
	}
	gi := NewGraphIndex()
	if count, err := LoadEdgesCSV(gi, paths, IngestOptions{}); err != nil || count != 1 {
		t.Fatalf("default options: count=%d error=%v", count, err)
	}
	ingestCheckGraph(t, gi, []Edge{{From: 2, To: 3}})
}

func BenchmarkLoadEdgesCSV(b *testing.B) {
	// A repeatable, densely connected graph keeps index memory bounded while
	// exercising parsing and real AddEdgesBatch calls (including deduplication).
	var block bytes.Buffer
	for i := 0; i < 65536; i++ {
		fmt.Fprintf(&block, "%d,%d,%d\n", 1000000000+i, i%256, (i/256)%256)
	}
	path := filepath.Join(b.TempDir(), "edges.csv")
	f, err := os.Create(path)
	if err != nil {
		b.Fatal(err)
	}
	var size, rows int64
	for size < 50<<20 {
		n, err := f.Write(block.Bytes())
		if err != nil {
			f.Close()
			b.Fatal(err)
		}
		size += int64(n)
		rows += 65536
	}
	if err := f.Close(); err != nil {
		b.Fatal(err)
	}
	for _, workers := range []int{1, runtime.NumCPU()} {
		b.Run(fmt.Sprintf("workers_%d", workers), func(b *testing.B) {
			b.SetBytes(size)
			b.ReportAllocs()
			chunk := max(int64(256<<10), size/int64(workers*4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				gi := NewGraphIndex()
				count, err := LoadEdgesCSV(gi, []string{path}, IngestOptions{
					Workers: workers, ChunkBytes: chunk, BatchEdges: 1 << 16,
				})
				if err != nil || count != rows {
					b.Fatalf("count=%d error=%v, want %d", count, err, rows)
				}
			}
		})
	}
}
