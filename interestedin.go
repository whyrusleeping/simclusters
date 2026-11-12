package simclusters

import (
	"bufio"
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
)

// communityScore is one entry of a user's or influencer's community vector.
type communityScore struct {
	Community int
	Score     float64
}

// interestIndex holds the community scores of the clustered influencers by
// dense user index.
type interestIndex struct {
	gi             *GraphIndex
	slot           []int32            // dense user -> index into scores, or -1
	scores         [][]communityScore // ascending community
	numCommunities int                // largest community + 1
}

func newInterestIndex(gi *GraphIndex, knownFor map[int64]map[int]float64, workers int) *interestIndex {
	gi.build(workers)
	x := &interestIndex{gi: gi, slot: make([]int32, len(gi.ids))}
	parallelRange(len(x.slot), workers, 1<<16, func(_, lo, hi int) {
		for u := lo; u < hi; u++ {
			x.slot[u] = -1
		}
	})
	for id, comms := range knownFor {
		u, ok := gi.index(id)
		if !ok {
			continue
		}
		list := make([]communityScore, 0, len(comms))
		for c, s := range comms {
			// Scores are positive; interestedIn relies on it.
			if s > 0 && c >= 0 {
				list = append(list, communityScore{c, s})
				x.numCommunities = max(x.numCommunities, c+1)
			}
		}
		// A fixed order keeps the floating-point sums, and so the output,
		// identical from run to run.
		slices.SortFunc(list, func(a, b communityScore) int { return cmp.Compare(a.Community, b.Community) })
		x.slot[u] = int32(len(x.scores))
		x.scores = append(x.scores, list)
	}
	return x
}

// interestedIn computes dense user u's community interests from the
// influencers they follow: the scores of all followed influencers are summed
// per community and normalized to sum to 1, and the topN largest are
// returned in descending order. acc must have numCommunities zeros and is
// left that way.
func (x *interestIndex) interestedIn(u uint32, topN int, acc []float64, out []communityScore) []communityScore {
	var total float64
	out = out[:0]
	for _, v := range x.gi.row(u) {
		s := x.slot[v]
		if s < 0 {
			continue
		}
		for _, cs := range x.scores[s] {
			if acc[cs.Community] == 0 {
				out = append(out, communityScore{Community: cs.Community})
			}
			acc[cs.Community] += cs.Score
			total += cs.Score
		}
	}
	for i := range out {
		c := out[i].Community
		out[i].Score = acc[c] / total
		acc[c] = 0
	}
	slices.SortFunc(out, func(a, b communityScore) int {
		return cmp.Or(cmp.Compare(b.Score, a.Score), cmp.Compare(a.Community, b.Community))
	})
	if len(out) > topN {
		out = out[:topN]
	}
	return out
}

// writeInterestedIn computes the interests of every user that follows at
// least one clustered influencer and streams them to w as one JSON object,
// {"<user id>": {"<community>": <score>, ...}, ...}, ordered by user id. It
// returns the number of users written.
func writeInterestedIn(w io.Writer, gi *GraphIndex, knownFor map[int64]map[int]float64, topN, workers int) (int, error) {
	x := newInterestIndex(gi, knownFor, workers)
	numUsers := len(gi.ids)

	// Users are encoded in parallel in fixed-size chunks; chunks are written
	// in order one wave at a time, which bounds memory to a wave of output.
	const chunkUsers = 4096
	numChunks := (numUsers + chunkUsers - 1) / chunkUsers
	wave := max(workers, 1) * 4
	bufs := make([][]byte, wave)
	counts := make([]int, wave)

	bw := bufio.NewWriterSize(w, 1<<20)
	bw.WriteByte('{')
	written := 0
	for first := 0; first < numChunks; first += wave {
		chunks := min(wave, numChunks-first)
		parallelRange(chunks, workers, 1, func(_, lo, hi int) {
			acc := make([]float64, x.numCommunities)
			var scores []communityScore
			for ci := lo; ci < hi; ci++ {
				buf := bufs[ci][:0]
				counts[ci] = 0
				start := (first + ci) * chunkUsers
				for u := start; u < min(start+chunkUsers, numUsers); u++ {
					scores = x.interestedIn(uint32(u), topN, acc, scores)
					if len(scores) == 0 {
						continue
					}
					buf = append(buf, ',', '"')
					buf = strconv.AppendInt(buf, gi.ids[u], 10)
					buf = append(buf, '"', ':', '{')
					for i, cs := range scores {
						if i > 0 {
							buf = append(buf, ',')
						}
						buf = append(buf, '"')
						buf = strconv.AppendInt(buf, int64(cs.Community), 10)
						buf = append(buf, '"', ':')
						buf = strconv.AppendFloat(buf, cs.Score, 'g', -1, 64)
					}
					buf = append(buf, '}')
					counts[ci]++
				}
				bufs[ci] = buf
			}
		})
		for ci := 0; ci < chunks; ci++ {
			buf := bufs[ci]
			if written == 0 && len(buf) > 0 {
				buf = buf[1:] // no comma before the first entry
			}
			if _, err := bw.Write(buf); err != nil {
				return written, err
			}
			written += counts[ci]
		}
	}
	bw.WriteString("}\n")
	return written, bw.Flush()
}

// writeFileAtomic writes a file through a temporary sibling and renames it
// into place, so a failed run never leaves a truncated result behind.
func writeFileAtomic(path string, write func(io.Writer) error) (err error) {
	f, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()
	if err = write(f); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	if err = f.Chmod(0o644); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// writeJSONFile atomically writes v as JSON.
func writeJSONFile(path string, v any) error {
	return writeFileAtomic(path, func(w io.Writer) error {
		bw := bufio.NewWriterSize(w, 1<<20)
		if err := json.NewEncoder(bw).Encode(v); err != nil {
			return err
		}
		return bw.Flush()
	})
}
