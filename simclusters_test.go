package simclusters

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const (
	e2eGroups      = 12
	e2ePerGroup    = 15
	e2ePoolSize    = 400
	e2eInfluencers = e2eGroups * e2ePerGroup
	e2eUserBase    = 10_000
)

// e2eFollowGraph builds a follow graph of groups of influencers (ids from 1)
// that share an audience: every influencer is followed by 60% of its group's
// pool of users, and every user also follows two random influencers.
func e2eFollowGraph() []Edge {
	r := rand.New(rand.NewSource(1))
	var edges []Edge
	for inf := 0; inf < e2eInfluencers; inf++ {
		pool := e2eUserBase + (inf/e2ePerGroup)*e2ePoolSize
		for u := 0; u < e2ePoolSize; u++ {
			if r.Float64() < 0.6 {
				edges = append(edges, Edge{From: int64(pool + u), To: int64(inf + 1)})
			}
		}
	}
	for u := 0; u < e2eGroups*e2ePoolSize; u++ {
		for i := 0; i < 2; i++ {
			edges = append(edges, Edge{From: int64(e2eUserBase + u), To: int64(r.Intn(e2eInfluencers) + 1)})
		}
	}
	return edges
}

func e2eOptions(t *testing.T) Options {
	opts := DefaultOptions()
	opts.MinFollowers = 50
	opts.NumCommunities = 2 * e2eGroups
	opts.Epochs = 30
	opts.Workers = 4
	opts.OutputDir = t.TempDir()
	return opts
}

func readAssignments(t *testing.T, path string) map[int64]map[int]float64 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var out map[int64]map[int]float64
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return out
}

func TestComputeClustersEndToEnd(t *testing.T) {
	opts := e2eOptions(t)
	opts.CheckpointPath = filepath.Join(t.TempDir(), "checkpoint")
	sc, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	sc.Graph().AddEdgesBatch(e2eFollowGraph())
	if err := sc.ComputeClusters(); err != nil {
		t.Fatal(err)
	}

	influencers := readAssignments(t, filepath.Join(opts.OutputDir, InfluencerCommunitiesFile))
	if len(influencers) != e2eInfluencers {
		t.Fatalf("%d influencers clustered, want %d", len(influencers), e2eInfluencers)
	}
	groupComms := make([]map[int]bool, e2eGroups)
	for id, comms := range influencers {
		if len(comms) < 1 || len(comms) > opts.MaxCommunitiesPerNode {
			t.Fatalf("influencer %d is in %d communities", id, len(comms))
		}
		g := int(id-1) / e2ePerGroup
		if groupComms[g] == nil {
			groupComms[g] = make(map[int]bool)
		}
		for c := range comms {
			groupComms[g][c] = true
		}
	}
	share := func(a, b int64) bool {
		for c := range influencers[a] {
			if _, ok := influencers[b][c]; ok {
				return true
			}
		}
		return false
	}
	var same, sameShared, cross, crossShared int
	for a := int64(1); a <= e2eInfluencers; a++ {
		for b := a + 1; b <= e2eInfluencers; b++ {
			if (a-1)/e2ePerGroup == (b-1)/e2ePerGroup {
				same++
				if share(a, b) {
					sameShared++
				}
			} else {
				cross++
				if share(a, b) {
					crossShared++
				}
			}
		}
	}
	if float64(sameShared) < 0.9*float64(same) || float64(crossShared) > 0.05*float64(cross) {
		t.Errorf("influencers sharing a community: %d/%d within groups, %d/%d across groups", sameShared, same, crossShared, cross)
	}

	users := readAssignments(t, filepath.Join(opts.OutputDir, MappingFile))
	if want := e2eGroups * e2ePoolSize; len(users) != want {
		t.Fatalf("%d users mapped, want %d", len(users), want)
	}
	topInOwnGroup := 0
	for id, comms := range users {
		if len(comms) < 1 || len(comms) > opts.TopInterests {
			t.Fatalf("user %d has %d communities", id, len(comms))
		}
		top, best, sum := -1, 0.0, 0.0
		for c, score := range comms {
			sum += score
			if score > best || (score == best && c < top) {
				top, best = c, score
			}
		}
		if sum > 1+1e-9 {
			t.Fatalf("user %d scores sum to %v", id, sum)
		}
		if groupComms[int(id-e2eUserBase)/e2ePoolSize][top] {
			topInOwnGroup++
		}
	}
	if float64(topInOwnGroup) < 0.95*float64(len(users)) {
		t.Errorf("only %d/%d users have a top community from their own group", topInOwnGroup, len(users))
	}

	// Resuming from the checkpoint and changing the worker count must both
	// reproduce the fresh run exactly.
	fresh := readOutputs(t, opts.OutputDir)

	resumeOpts := e2eOptions(t)
	resumed, err := New(resumeOpts)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.ComputeClustersFromCheckpoint(opts.CheckpointPath); err != nil {
		t.Fatal(err)
	}
	if !slices.EqualFunc(fresh, readOutputs(t, resumeOpts.OutputDir), bytes.Equal) {
		t.Error("resuming from the checkpoint gave different output than the fresh run")
	}

	serialOpts := e2eOptions(t)
	serialOpts.Workers = 1
	serial, err := New(serialOpts)
	if err != nil {
		t.Fatal(err)
	}
	serial.Graph().AddEdgesBatch(e2eFollowGraph())
	if err := serial.ComputeClusters(); err != nil {
		t.Fatal(err)
	}
	if !slices.EqualFunc(fresh, readOutputs(t, serialOpts.OutputDir), bytes.Equal) {
		t.Error("a different worker count gave different output")
	}
}

func readOutputs(t *testing.T, dir string) [][]byte {
	t.Helper()
	var out [][]byte
	for _, name := range []string{InfluencerCommunitiesFile, MappingFile} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, data)
	}
	return out
}

func TestResumeRejectsCheckpointWithHigherFloor(t *testing.T) {
	opts := e2eOptions(t)
	opts.MinSimilarity = 0.4 // and so is the checkpoint's floor
	opts.CheckpointPath = filepath.Join(t.TempDir(), "checkpoint")
	sc, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	sc.Graph().AddEdgesBatch(e2eFollowGraph())
	if err := sc.ComputeClusters(); err != nil {
		t.Fatal(err)
	}

	resumeOpts := e2eOptions(t) // MinSimilarity 0.2, below the checkpoint's floor
	resumed, err := New(resumeOpts)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.ComputeClustersFromCheckpoint(opts.CheckpointPath); err == nil {
		t.Error("resuming with MinSimilarity below the checkpoint's floor succeeded")
	}
}

func TestComputeClustersWithoutInfluencers(t *testing.T) {
	opts := e2eOptions(t)
	sc, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	if err := sc.ComputeClusters(); err == nil {
		t.Error("empty graph: expected an error")
	}

	sc.Graph().AddEdgesBatch([]Edge{{From: 1, To: 2}, {From: 3, To: 2}})
	if err := sc.ComputeClusters(); err == nil {
		t.Error("graph without influencers: expected an error")
	}
	for _, name := range []string{InfluencerCommunitiesFile, MappingFile} {
		if _, err := os.Stat(filepath.Join(opts.OutputDir, name)); err == nil {
			t.Errorf("%s was written by a failed run", name)
		}
	}
}

func TestIdentifyInfluencers(t *testing.T) {
	opts := DefaultOptions()
	opts.MinFollowers = 3
	sc, err := New(opts)
	if err != nil {
		t.Fatal(err)
	}
	// User 1 has 4 followers, user 2 has 3, user 3 has 2; users 10..13 have
	// none. Of these 7 users the 90th percentile is 4 followers, which is
	// above MinFollowers and so sets the cut.
	var edges []Edge
	for f := int64(10); f < 14; f++ {
		edges = append(edges, Edge{From: f, To: 1}, Edge{From: f, To: 1}) // duplicates count once
		if f < 13 {
			edges = append(edges, Edge{From: f, To: 2})
		}
		if f < 12 {
			edges = append(edges, Edge{From: f, To: 3})
		}
	}
	sc.Graph().AddEdgesBatch(edges)
	if got := sc.identifyInfluencers(); !slices.Equal(got, []int64{1}) {
		t.Errorf("influencers = %v, want [1]", got)
	}

	// With 40 more users that have no followers the 90th percentile drops
	// to zero and MinFollowers decides.
	edges = edges[:0]
	for f := int64(100); f < 140; f++ {
		edges = append(edges, Edge{From: f, To: 50})
	}
	sc.Graph().AddEdgesBatch(edges)
	if got := sc.identifyInfluencers(); !slices.Equal(got, []int64{1, 2, 50}) {
		t.Errorf("influencers = %v, want [1 2 50]", got)
	}
}

func TestOptionsValidate(t *testing.T) {
	if _, err := New(DefaultOptions()); err != nil {
		t.Fatalf("default options rejected: %v", err)
	}
	for name, mutate := range map[string]func(*Options){
		"floor above min similarity": func(o *Options) { o.SimilarityFloor = o.MinSimilarity * 2 },
		"too many per node":          func(o *Options) { o.MaxCommunitiesPerNode, o.MaxCandidates = maxCandidateLimit, maxCandidateLimit },
		"no room for a candidate":    func(o *Options) { o.MaxCandidates = o.MaxCommunitiesPerNode },
		"no communities":             func(o *Options) { o.NumCommunities = 0 },
		"zero temperature":           func(o *Options) { o.Temperature = 0 },
		"NaN alpha":                  func(o *Options) { o.Alpha = math.NaN() },
		"infinite temperature":       func(o *Options) { o.Temperature = math.Inf(1) },
	} {
		opts := DefaultOptions()
		mutate(&opts)
		if _, err := New(opts); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
