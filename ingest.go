package simclusters

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
)

// IngestOptions tunes LoadEdgesCSV. Zero values select defaults.
type IngestOptions struct {
	Workers    int               // Parallel parsers; default runtime.NumCPU().
	ChunkBytes int64             // Approximate newline-aligned range size; default 64 MiB.
	BatchEdges int               // Edges buffered per worker; default 1<<16.
	Progress   func(bytes int64) // Optional concurrent callback reporting byte deltas.
}

// LoadEdgesCSV parses the files and adds every edge to gi. It returns the number
// of edges loaded, counting duplicate rows. On error, previously flushed batches
// remain in gi and the returned count describes those batches. Progress must be
// safe for concurrent calls. Input files must not change during the load.
func LoadEdgesCSV(gi *GraphIndex, paths []string, opts IngestOptions) (int64, error) {
	if len(paths) == 0 {
		return 0, errors.New("LoadEdgesCSV: no input paths")
	}
	if gi == nil {
		return 0, errors.New("LoadEdgesCSV: nil graph index")
	}
	if opts.Workers < 0 || opts.ChunkBytes < 0 || opts.BatchEdges < 0 {
		return 0, errors.New("LoadEdgesCSV: options must be non-negative")
	}
	if opts.Workers == 0 {
		opts.Workers = runtime.NumCPU()
	}
	if opts.ChunkBytes == 0 {
		opts.ChunkBytes = 64 << 20
	}
	if opts.BatchEdges == 0 {
		opts.BatchEdges = 1 << 16
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var firstErr error
	var once sync.Once
	fail := func(err error) {
		once.Do(func() {
			firstErr = err
			cancel()
		})
	}
	var loaded atomic.Int64
	jobs := make(chan ingestRange)
	var wg sync.WaitGroup
	for i := 0; i < opts.Workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Allocate lazily: idle workers need neither a batch nor a read buffer.
			var batch []Edge
			var reader *bufio.Reader
			flush := func() {
				if len(batch) != 0 {
					gi.AddEdgesBatch(batch)
					loaded.Add(int64(len(batch)))
					batch = batch[:0]
				}
			}
			for {
				select {
				case <-ctx.Done():
					return
				case job, ok := <-jobs:
					if !ok {
						if ctx.Err() == nil {
							flush()
						}
						return
					}
					if ctx.Err() != nil {
						return
					}
					if reader == nil {
						reader = bufio.NewReaderSize(nil, 256<<10)
						batch = make([]Edge, 0, opts.BatchEdges)
					}
					err := ingestReadRange(ctx, job, reader, opts.Progress, func(edge Edge) {
						batch = append(batch, edge)
						if len(batch) == cap(batch) {
							flush()
						}
					})
					if err != nil {
						if !errors.Is(err, context.Canceled) {
							fail(err)
						}
						return
					}
				}
			}
		}()
	}

	// Only the scheduler reads boundary bytes. In particular, a line longer than
	// a chunk is scanned once, not once for every nominal boundary it crosses.
	boundaryBuf := make([]byte, 64<<10)
	for _, path := range paths {
		if ctx.Err() != nil {
			break
		}
		if err := ingestSchedule(ctx, path, opts.ChunkBytes, boundaryBuf, jobs); err != nil {
			if !errors.Is(err, context.Canceled) {
				fail(err)
			}
			break
		}
	}
	close(jobs)
	wg.Wait()
	return loaded.Load(), firstErr
}

type ingestRange struct {
	path       string
	start, end int64
}

func ingestError(path string, offset int64, err error) error {
	return fmt.Errorf("%s: byte offset %d: %w", path, offset, err)
}

func ingestSchedule(ctx context.Context, path string, chunkBytes int64, buf []byte, jobs chan<- ingestRange) error {
	f, err := os.Open(path)
	if err != nil {
		return ingestError(path, 0, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ingestError(path, 0, err)
	}
	if !info.Mode().IsRegular() {
		return ingestError(path, 0, errors.New("input is not a regular file"))
	}
	size := info.Size()
	for start := int64(0); start < size; {
		end := size
		if chunkBytes < size-start {
			// Starting one byte before the proposed boundary handles the case
			// where that boundary is already the first byte of a line.
			pos := start + chunkBytes - 1
			for pos < size {
				if err := ctx.Err(); err != nil {
					return err
				}
				nwant := int64(len(buf))
				if nwant > size-pos {
					nwant = size - pos
				}
				n, readErr := f.ReadAt(buf[:nwant], pos)
				if readErr != nil {
					return ingestError(path, start, readErr)
				}
				if i := bytes.IndexByte(buf[:n], '\n'); i >= 0 {
					end = pos + int64(i) + 1
					break
				}
				pos += int64(n)
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case jobs <- ingestRange{path: path, start: start, end: end}:
		}
		start = end
	}
	return nil
}

type ingestProgressReader struct {
	reader   *io.SectionReader
	progress func(int64)
}

func (r ingestProgressReader) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if n > 0 && r.progress != nil {
		r.progress(int64(n))
	}
	return n, err
}

func ingestReadRange(ctx context.Context, job ingestRange, reader *bufio.Reader, progress func(int64), add func(Edge)) error {
	f, err := os.Open(job.path)
	if err != nil {
		return ingestError(job.path, job.start, err)
	}
	defer f.Close()
	reader.Reset(ingestProgressReader{io.NewSectionReader(f, job.start, job.end-job.start), progress})
	lineOffset, consumed := job.start, job.start
	var row ingestCSVRow
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		part, readErr := reader.ReadSlice('\n')
		consumed += int64(len(part))
		if readErr != nil && readErr != bufio.ErrBufferFull && readErr != io.EOF {
			return ingestError(job.path, lineOffset, readErr)
		}
		if len(part) > 0 && part[len(part)-1] == '\n' {
			part = part[:len(part)-1]
		}
		row.consume(part)
		if readErr == bufio.ErrBufferFull {
			continue
		}
		if readErr == io.EOF && consumed != job.end {
			return ingestError(job.path, lineOffset, io.ErrUnexpectedEOF)
		}
		edge, present, err := row.finish(lineOffset == 0)
		if err != nil {
			return ingestError(job.path, lineOffset, err)
		}
		if present {
			add(edge)
		}
		if readErr == io.EOF {
			return nil
		}
		row = ingestCSVRow{}
		lineOffset = consumed
	}
}

// A row is parsed incrementally so even arbitrarily long lines (including ignored
// extra fields) need no storage beyond the worker's fixed-size read buffer.
type ingestCSVRow struct {
	numbers   [3]ingestCSVNumber
	field     int
	nonempty  bool
	pendingCR bool
}

type ingestCSVNumber struct {
	value      int64
	nonempty   bool
	invalid    bool
	nonnumeric bool
}

func (n *ingestCSVNumber) consume(data []byte) {
	if len(data) != 0 {
		n.nonempty = true
	}
	value := n.value
	for _, c := range data {
		digit := c - '0'
		if digit > 9 {
			n.invalid = true
			// Signs are malformed integers, not a header. Likewise, empty
			// fields and overflow must not silently turn the first row into one.
			if c != '-' && c != '+' {
				n.nonnumeric = true
			}
			continue
		}
		if value > (1<<63-1)/10 || value == (1<<63-1)/10 && digit > 7 {
			n.invalid = true
		} else {
			value = value*10 + int64(digit)
		}
	}
	n.value = value
}

func (r *ingestCSVRow) consume(data []byte) {
	if len(data) == 0 {
		return
	}
	if r.pendingCR {
		// A CR is allowed only immediately before the line ending (or EOF).
		r.nonempty = true
		if r.field < 3 {
			r.numbers[r.field].consume([]byte{'\r'})
		}
		r.pendingCR = false
	}
	if data[len(data)-1] == '\r' {
		r.pendingCR = true
		data = data[:len(data)-1]
	}
	if len(data) != 0 {
		r.nonempty = true
	}
	for len(data) > 0 && r.field < 3 {
		i := bytes.IndexByte(data, ',')
		if i < 0 {
			r.numbers[r.field].consume(data)
			return
		}
		r.numbers[r.field].consume(data[:i])
		r.field++
		data = data[i+1:]
	}
}

func (r *ingestCSVRow) finish(first bool) (Edge, bool, error) {
	if !r.nonempty {
		return Edge{}, false, nil
	}
	if r.field < 2 {
		return Edge{}, false, errors.New("row has fewer than three fields")
	}
	if first && (r.numbers[1].nonnumeric || r.numbers[2].nonnumeric) {
		return Edge{}, false, nil
	}
	for i := range r.numbers {
		n := &r.numbers[i]
		if !n.nonempty || n.invalid {
			return Edge{}, false, fmt.Errorf("field %d is not a non-negative int64", i+1)
		}
	}
	return Edge{From: r.numbers[1].value, To: r.numbers[2].value}, true, nil
}
