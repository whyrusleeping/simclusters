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

	"github.com/RoaringBitmap/roaring/v2/roaring64"
)

// communityScore is one entry of a user's or influencer's community vector.
type communityScore struct {
	Community int
	Score     float64
}

// interestedIn computes a user's community interests from the influencers
// they follow: the scores of all followed influencers are summed per
// community and normalized to sum to 1, and the topN largest are returned in
// descending order. acc is scratch space and is left empty.
func interestedIn(following *roaring64.Bitmap, knownFor map[int64][]communityScore, topN int, acc map[int]float64, out []communityScore) []communityScore {
	var total float64
	it := following.Iterator()
	for it.HasNext() {
		for _, cs := range knownFor[int64(it.Next())] {
			acc[cs.Community] += cs.Score
			total += cs.Score
		}
	}
	out = out[:0]
	for comm, score := range acc {
		out = append(out, communityScore{comm, score / total})
	}
	clear(acc)
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
	index := make(map[int64][]communityScore, len(knownFor))
	for id, comms := range knownFor {
		list := make([]communityScore, 0, len(comms))
		for c, s := range comms {
			list = append(list, communityScore{c, s})
		}
		// A fixed order keeps the floating-point sums, and so the output,
		// identical from run to run.
		slices.SortFunc(list, func(a, b communityScore) int { return cmp.Compare(a.Community, b.Community) })
		index[id] = list
	}

	var users []int64
	gi.Following.Range(func(user int64, _ *roaring64.Bitmap) bool {
		users = append(users, user)
		return true
	})
	slices.Sort(users)

	// Users are encoded in parallel in fixed-size chunks; chunks are written
	// in order one wave at a time, which bounds memory to a wave of output.
	const chunkUsers = 4096
	numChunks := (len(users) + chunkUsers - 1) / chunkUsers
	wave := max(workers, 1) * 4
	bufs := make([][]byte, wave)
	counts := make([]int, wave)

	bw := bufio.NewWriterSize(w, 1<<20)
	bw.WriteByte('{')
	written := 0
	for first := 0; first < numChunks; first += wave {
		chunks := min(wave, numChunks-first)
		parallelRange(chunks, workers, 1, func(_, lo, hi int) {
			acc := make(map[int]float64)
			var scores []communityScore
			for ci := lo; ci < hi; ci++ {
				buf := bufs[ci][:0]
				counts[ci] = 0
				start := (first + ci) * chunkUsers
				for _, user := range users[start:min(start+chunkUsers, len(users))] {
					scores = interestedIn(gi.GetFollowing(user), index, topN, acc, scores)
					if len(scores) == 0 {
						continue
					}
					buf = append(buf, ',', '"')
					buf = strconv.AppendInt(buf, user, 10)
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
