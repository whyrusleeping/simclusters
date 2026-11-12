package simclusters

import (
	"log/slog"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"time"
)

// Neighborhood-aware Metropolis-Hastings community detection, after
// Algorithms 1 and 3 of the SimClusters paper (KDD 2020).
//
// Z assigns every node of the similarity graph to between 1 and l communities.
// With edge weights w, the per-node objective (paper Eq. 2, weighted) is
//
//	f(u, s) = alpha * sum of w(u,v) over neighbours v sharing a community with s
//	        - number of non-neighbours sharing a community with s
//
// up to a constant. A neighbour or non-neighbour counts once no matter how
// many communities it shares.
//
// Each epoch visits every node once in shuffled order. Nodes are processed in
// small batches: proposals for a batch are computed in parallel against the
// state at the start of the batch and then applied in order. Together with
// per-(epoch, node) random streams this makes the result a function of the
// seed only, independent of the number of workers.

const (
	// maxCandidateLimit bounds candidate communities per step: candidate
	// subsets are bitmasks in a uint8.
	maxCandidateLimit = 8
	// initSeedNeighbors is how many of a seed node's strongest neighbours
	// join its community at initialization.
	initSeedNeighbors = 20
	// scoreEps absorbs floating-point noise when comparing objective values.
	scoreEps = 1e-9
)

type mhParams struct {
	numCommunities int     // k
	maxPerNode     int     // l, below maxCandidateLimit
	maxCandidates  int     // communities considered per step, in (maxPerNode, maxCandidateLimit]
	alpha          float64 // weight of covered edges relative to shared non-edges
	epsilon        float64 // probability of adding a uniformly random candidate community
	temperature    float64 // softmax/acceptance temperature in objective units
	seed           uint64
	workers        int
	batchSize      int // nodes proposed against the same snapshot; 0 picks a default
}

// mhState is the community assignment being optimized. z and members always
// describe the same assignment.
type mhState struct {
	g       *simGraph
	p       mhParams
	z       []int32   // n*l community ids; each row is packed and padded with -1
	members [][]int32 // community -> nodes
	masks   [][]uint8 // masks[nc]: non-empty subsets of nc candidates of size <= l, by size
	order   []int32
	props   []proposal
	scratch []*mhScratch
}

// mhScratch is per-worker working memory. Everything is left zeroed between
// steps.
type mhScratch struct {
	commW   []float64 // community -> weight of neighbours in it
	commCnt []int32   // community -> number of neighbours in it
	seen    []int32   // communities with commCnt != 0
	candBit []uint8   // community -> its bit among the current candidates
	mark    []uint8   // node -> mask of candidate communities it belongs to
	touched []int32   // nodes with mark != 0
	gain    [1 << maxCandidateLimit]float64
	cnt     [1 << maxCandidateLimit]int32
	f       [1 << maxCandidateLimit]float64
	cum     [1 << maxCandidateLimit]float64
}

type proposal struct {
	set    [maxCandidateLimit]int32
	n      int
	accept bool
	delta  float64
}

// stepRNG is a deterministic generator seeded per (seed, stream, index).
type stepRNG struct{ pcg rand.PCG }

func (r *stepRNG) seed(seed, stream, index uint64) {
	r.pcg.Seed(splitmix64(seed^splitmix64(stream)), splitmix64(index))
}

func (r *stepRNG) float64() float64 { return float64(r.pcg.Uint64()>>11) / (1 << 53) }

func (r *stepRNG) intn(n int) int { return int(r.pcg.Uint64() % uint64(n)) }

const (
	rngStreamInit = iota
	rngStreamShuffle
	rngStreamStep
	rngStreams
)

func newMHState(g *simGraph, p mhParams) *mhState {
	n := g.numNodes()
	p.maxPerNode = min(max(p.maxPerNode, 1), maxCandidateLimit-1)
	// One more than l, so that a node in l communities can still weigh another.
	p.maxCandidates = min(max(p.maxCandidates, p.maxPerNode+1), maxCandidateLimit)
	p.numCommunities = max(p.numCommunities, 1)
	p.workers = max(p.workers, 1)
	if p.temperature <= 0 {
		p.temperature = 1
	}
	if p.batchSize <= 0 {
		// Small enough that few of a node's neighbours move in the same
		// batch, and independent of the worker count so results are too.
		p.batchSize = min(max(n/64, 1), 1024)
	}

	s := &mhState{
		g:       g,
		p:       p,
		z:       make([]int32, n*p.maxPerNode),
		members: make([][]int32, p.numCommunities),
		masks:   make([][]uint8, p.maxCandidates+1),
		order:   make([]int32, n),
		props:   make([]proposal, p.batchSize),
		scratch: make([]*mhScratch, p.workers),
	}
	for nc := 1; nc <= p.maxCandidates; nc++ {
		for size := 1; size <= p.maxPerNode; size++ {
			for m := 1; m < 1<<nc; m++ {
				if bits.OnesCount(uint(m)) == size {
					s.masks[nc] = append(s.masks[nc], uint8(m))
				}
			}
		}
	}
	for i := range s.scratch {
		s.scratch[i] = &mhScratch{
			commW:   make([]float64, p.numCommunities),
			commCnt: make([]int32, p.numCommunities),
			candBit: make([]uint8, p.numCommunities),
			mark:    make([]uint8, n),
		}
	}
	s.initialize()
	return s
}

// row returns node u's communities: valid ids first, then -1 padding.
func (s *mhState) row(u int32) []int32 {
	l := s.p.maxPerNode
	return s.z[int(u)*l : int(u)*l+l]
}

// communities returns node u's communities without padding.
func (s *mhState) communities(u int32) []int32 {
	row := s.row(u)
	for i, c := range row {
		if c < 0 {
			return row[:i]
		}
	}
	return row
}

// initialize seeds each community with a random node and its strongest
// neighbours (paper Algorithm 3), then trims every node to its l best
// communities so the per-node cap holds from the start.
func (s *mhState) initialize() {
	n, k, l := s.g.numNodes(), s.p.numCommunities, s.p.maxPerNode
	if n == 0 {
		return
	}
	var rng stepRNG
	rng.seed(s.p.seed, rngStreamInit, 0)

	perm := make([]int32, n)
	for i := range perm {
		perm[i] = int32(i)
	}
	for i := n - 1; i > 0; i-- {
		j := rng.intn(i + 1)
		perm[i], perm[j] = perm[j], perm[i]
	}

	type seeded struct {
		comm     int32
		affinity float32 // weight of the edge to the seed node
	}
	lists := make([][]seeded, n)
	for c := 0; c < k; c++ {
		seed := perm[c%n]
		lists[seed] = append(lists[seed], seeded{int32(c), math.MaxFloat32})
		nbrs, ws := s.g.neighbors(seed)
		for i := 0; i < len(nbrs) && i < initSeedNeighbors; i++ {
			lists[nbrs[i]] = append(lists[nbrs[i]], seeded{int32(c), ws[i]})
		}
	}

	for u := range lists {
		list := lists[u]
		slices.SortFunc(list, func(a, b seeded) int {
			if a.affinity != b.affinity {
				if a.affinity > b.affinity {
					return -1
				}
				return 1
			}
			return int(a.comm - b.comm)
		})
		if len(list) > l {
			list = list[:l]
		}
		row := s.row(int32(u))
		for i := range row {
			row[i] = -1
		}
		if len(list) == 0 {
			list = append(list, seeded{comm: int32(rng.intn(k))})
		}
		for i, sd := range list {
			row[i] = sd.comm
			s.members[sd.comm] = append(s.members[sd.comm], int32(u))
		}
	}
}

// propose evaluates node u against the current state and returns the
// community set it should move to, if any. It only reads shared state.
// A temperature of zero selects the best candidate set deterministically.
func (s *mhState) propose(u int32, sc *mhScratch, rng *stepRNG, temperature float64) proposal {
	p := &s.p
	greedy := temperature <= 0
	old := s.communities(u)
	nbrs, ws := s.g.neighbors(u)
	l := p.maxPerNode

	// Per-community neighbour weight, to rank the communities seen in the
	// neighbourhood.
	for i, v := range nbrs {
		w := float64(ws[i])
		for _, c := range s.z[int(v)*l : int(v)*l+l] {
			if c < 0 {
				break
			}
			if sc.commCnt[c] == 0 {
				sc.seen = append(sc.seen, c)
			}
			sc.commW[c] += w
			sc.commCnt[c]++
		}
	}

	// Candidates: the current communities, then the neighbourhood communities
	// with the best single-community score.
	var cands [maxCandidateLimit]int32
	nOld := copy(cands[:], old)
	slots := p.maxCandidates - nOld
	var topC [maxCandidateLimit]int32
	var topS [maxCandidateLimit]float64
	nTop := 0
	for _, c := range sc.seen {
		if slices.Contains(old, c) {
			continue
		}
		score := p.alpha*sc.commW[c] - float64(int32(len(s.members[c]))-sc.commCnt[c])
		pos := nTop
		for pos > 0 && (score > topS[pos-1] || (score == topS[pos-1] && c < topC[pos-1])) {
			pos--
		}
		if pos >= slots {
			continue
		}
		if nTop < slots {
			nTop++
		}
		copy(topC[pos+1:nTop], topC[pos:nTop-1])
		copy(topS[pos+1:nTop], topS[pos:nTop-1])
		topC[pos], topS[pos] = c, score
	}
	for _, c := range sc.seen {
		sc.commW[c], sc.commCnt[c] = 0, 0
	}
	sc.seen = sc.seen[:0]
	nc := nOld + copy(cands[nOld:], topC[:nTop])

	// Exploration: occasionally consider a uniformly random community, in
	// place of the weakest neighbourhood candidate if there is no free slot.
	if !greedy {
		explore := rng.float64() < p.epsilon
		r := int32(rng.intn(p.numCommunities))
		if explore && !slices.Contains(cands[:nc], r) {
			if nc < p.maxCandidates {
				cands[nc] = r
				nc++
			} else if nc > nOld {
				cands[nc-1] = r
			}
		}
	}
	if nc == 0 {
		return proposal{}
	}

	// For every node sharing a candidate community with u, find the mask of
	// candidates it belongs to. gain[m] and cnt[m] accumulate, over nodes with
	// mask m, alpha * edge weight and the number of non-neighbours.
	full := 1<<nc - 1
	gain, cnt, f := sc.gain[:full+1], sc.cnt[:full+1], sc.f[:full+1]
	clear(gain)
	clear(cnt)
	for i := 0; i < nc; i++ {
		sc.candBit[cands[i]] = 1 << i
	}
	for i, v := range nbrs {
		var m uint8
		for _, c := range s.z[int(v)*l : int(v)*l+l] {
			if c < 0 {
				break
			}
			m |= sc.candBit[c]
		}
		if m != 0 {
			gain[m] += p.alpha * float64(ws[i])
			cnt[m]-- // a neighbour, so not a non-neighbour below
		}
	}
	for i := 0; i < nc; i++ {
		bit := uint8(1) << i
		for _, v := range s.members[cands[i]] {
			if v == u {
				continue
			}
			if sc.mark[v] == 0 {
				sc.touched = append(sc.touched, v)
			}
			sc.mark[v] |= bit
		}
		sc.candBit[cands[i]] = 0
	}
	for _, v := range sc.touched {
		cnt[sc.mark[v]]++
		sc.mark[v] = 0
	}
	sc.touched = sc.touched[:0]

	// A node with mask m counts towards f(u, s) iff m intersects s, so
	// f(u, s) = total - sum of contributions over masks inside the complement
	// of s. The subset sums come from a zeta transform.
	f[0] = 0
	for m := 1; m <= full; m++ {
		f[m] = gain[m] - float64(cnt[m])
	}
	for b := 0; b < nc; b++ {
		bit := 1 << b
		for m := bit; m <= full; m++ {
			if m&bit != 0 {
				f[m] += f[m^bit]
			}
		}
	}
	total := f[full]
	value := func(m uint8) float64 { return total - f[full^int(m)] }

	oldMask := uint8(1<<nOld - 1)
	fOld := value(oldMask)
	masks := s.masks[nc]

	var chosen uint8
	var delta float64
	accept := false
	if greedy {
		// Best subset; ties go to the smaller set, which drops memberships
		// that contribute nothing.
		best := math.Inf(-1)
		for _, m := range masks {
			if v := value(m); v > best+scoreEps {
				best, chosen = v, m
			}
		}
		delta = best - fOld
		accept = delta > scoreEps || (delta > -scoreEps && bits.OnesCount8(chosen) < nOld)
	} else {
		// Sample a subset with probability proportional to exp(f/T), then
		// apply the Metropolis-Hastings acceptance test.
		beta := 1 / temperature
		fMax := math.Inf(-1)
		for _, m := range masks {
			fMax = max(fMax, value(m))
		}
		cum := sc.cum[:len(masks)]
		var sum float64
		for i, m := range masks {
			sum += math.Exp(beta * (value(m) - fMax))
			cum[i] = sum
		}
		r := rng.float64() * sum
		idx := 0
		for idx < len(masks)-1 && cum[idx] <= r {
			idx++
		}
		chosen = masks[idx]
		delta = value(chosen) - fOld
		accept = delta >= 0 || rng.float64() < math.Exp(beta*delta)
	}
	if !accept || chosen == oldMask {
		return proposal{}
	}

	pr := proposal{accept: true, delta: delta}
	for i := 0; i < nc; i++ {
		if chosen>>i&1 != 0 {
			pr.set[pr.n] = cands[i]
			pr.n++
		}
	}
	return pr
}

// apply moves node u to newSet, keeping z and members in sync.
func (s *mhState) apply(u int32, newSet []int32) {
	row := s.row(u)
	for _, c := range row {
		if c < 0 {
			break
		}
		if !slices.Contains(newSet, c) {
			m := s.members[c]
			i := slices.Index(m, u)
			m[i] = m[len(m)-1]
			s.members[c] = m[:len(m)-1]
		}
	}
	for _, c := range newSet {
		if !slices.Contains(row, c) {
			s.members[c] = append(s.members[c], u)
		}
	}
	n := copy(row, newSet)
	for i := n; i < len(row); i++ {
		row[i] = -1
	}
}

type epochStats struct {
	moved int     // nodes whose community set changed
	gain  float64 // sum of the objective deltas the moves were accepted on
}

// runEpoch visits every node once. At temperature zero each node moves to its
// best candidate set if that improves the objective.
func (s *mhState) runEpoch(epoch int, temperature float64) epochStats {
	n := s.g.numNodes()
	var rng stepRNG
	rng.seed(s.p.seed, uint64(epoch)*rngStreams+rngStreamShuffle, 0)
	for i := range s.order {
		s.order[i] = int32(i)
	}
	for i := n - 1; i > 0; i-- {
		j := rng.intn(i + 1)
		s.order[i], s.order[j] = s.order[j], s.order[i]
	}

	var stats epochStats
	stepStream := uint64(epoch)*rngStreams + rngStreamStep
	for start := 0; start < n; start += s.p.batchSize {
		nodes := s.order[start:min(start+s.p.batchSize, n)]
		props := s.props[:len(nodes)]
		parallelRange(len(nodes), s.p.workers, 1, func(worker, lo, hi int) {
			sc := s.scratch[worker]
			var rng stepRNG
			for i := lo; i < hi; i++ {
				rng.seed(s.p.seed, stepStream, uint64(nodes[i]))
				props[i] = s.propose(nodes[i], sc, &rng, temperature)
			}
		})
		for i := range props {
			if props[i].accept {
				s.apply(nodes[i], props[i].set[:props[i].n])
				stats.moved++
				stats.gain += props[i].delta
			}
		}
	}
	return stats
}

// objectiveStats describes how well the assignment fits the graph.
type objectiveStats struct {
	// Score is alpha*CoveredWeight - SharedNonEdges: the paper's F(Z) minus
	// the constant number of non-adjacent pairs.
	Score          float64
	CoveredWeight  float64 // weight of edges whose endpoints share a community
	TotalWeight    float64 // weight of all edges
	CoveredEdges   int64
	SharedNonEdges int64 // non-adjacent pairs that share a community
}

func (s *mhState) objective() objectiveStats {
	n := s.g.numNodes()
	type acc struct {
		coveredW, totalW float64
		coveredE, shared int64
		_                [32]byte // keep workers off each other's cache lines
	}
	accs := make([]acc, s.p.workers)
	parallelRange(n, s.p.workers, 64, func(worker, lo, hi int) {
		sc, a := s.scratch[worker], &accs[worker]
		for u := int32(lo); u < int32(hi); u++ {
			for _, c := range s.communities(u) {
				for _, v := range s.members[c] {
					if v != u && sc.mark[v] == 0 {
						sc.mark[v] = 1
						sc.touched = append(sc.touched, v)
					}
				}
			}
			a.shared += int64(len(sc.touched))
			nbrs, ws := s.g.neighbors(u)
			for i, v := range nbrs {
				a.totalW += float64(ws[i])
				if sc.mark[v] != 0 {
					a.coveredW += float64(ws[i])
					a.coveredE++
				}
			}
			for _, v := range sc.touched {
				sc.mark[v] = 0
			}
			sc.touched = sc.touched[:0]
		}
	})

	// Every pair was counted from both ends.
	var o objectiveStats
	var shared int64
	for i := range accs {
		o.CoveredWeight += accs[i].coveredW / 2
		o.TotalWeight += accs[i].totalW / 2
		o.CoveredEdges += accs[i].coveredE
		shared += accs[i].shared
	}
	o.CoveredEdges /= 2
	o.SharedNonEdges = shared/2 - o.CoveredEdges
	o.Score = s.p.alpha*o.CoveredWeight - float64(o.SharedNonEdges)
	return o
}

// assignments returns, for every node, its communities scored by the share of
// the node's similarity weight that goes to members of that community.
// Communities in which a node has no neighbour are dropped unless it has no
// better one.
func (s *mhState) assignments() map[int64]map[int]float64 {
	n, l := s.g.numNodes(), s.p.maxPerNode
	perNode := make([]map[int]float64, n)
	parallelRange(n, s.p.workers, 64, func(_, lo, hi int) {
		affinity := make([]float64, l)
		for u := int32(lo); u < int32(hi); u++ {
			comms := s.communities(u)
			clear(affinity)
			nbrs, ws := s.g.neighbors(u)
			for i, v := range nbrs {
				for _, c := range s.communities(v) {
					if j := slices.Index(comms, c); j >= 0 {
						affinity[j] += float64(ws[i])
					}
				}
			}
			scores := make(map[int]float64, len(comms))
			for j, c := range comms {
				if affinity[j] > 0 {
					scores[int(c)] = affinity[j] / s.g.rowW[u]
				}
			}
			if len(scores) == 0 {
				for _, c := range comms {
					scores[int(c)] = 1 / float64(len(comms))
				}
			}
			perNode[u] = scores
		}
	})
	out := make(map[int64]map[int]float64, n)
	for u, scores := range perNode {
		out[s.g.ids[u]] = scores
	}
	return out
}

// detectCommunities runs epochs of Metropolis-Hastings sampling followed by
// greedyEpochs of deterministic improvement.
func detectCommunities(g *simGraph, p mhParams, epochs, greedyEpochs int) *mhState {
	start := time.Now()
	s := newMHState(g, p)
	slog.Info("community detection started",
		"nodes", g.numNodes(), "edges", g.numEdges,
		"communities", s.p.numCommunities, "max_per_node", s.p.maxPerNode,
		"alpha", s.p.alpha, "temperature", s.p.temperature, "seed", s.p.seed,
		"workers", s.p.workers, "batch", s.p.batchSize)
	s.logObjective("initial assignment", 0, epochStats{})

	total := epochs + greedyEpochs
	logEvery := max(total/20, 1)
	for e := 1; e <= total; e++ {
		greedy := e > epochs
		temperature := s.p.temperature
		if greedy {
			temperature = 0
		}
		stats := s.runEpoch(e, temperature)
		if e%logEvery == 0 || e == epochs || e == total {
			msg := "epoch"
			if greedy {
				msg = "greedy epoch"
			}
			s.logObjective(msg, e, stats)
		}
	}
	slog.Info("community detection finished", "took", time.Since(start).Round(time.Millisecond))
	return s
}

// logCommunitySizes summarizes how nodes are spread over communities.
func (s *mhState) logCommunitySizes() {
	var sizes []int
	memberships := 0
	for _, m := range s.members {
		if len(m) > 0 {
			sizes = append(sizes, len(m))
			memberships += len(m)
		}
	}
	if len(sizes) == 0 {
		return
	}
	slices.Sort(sizes)
	slog.Info("community sizes",
		"non_empty", len(sizes), "of", len(s.members),
		"median", sizes[len(sizes)/2], "p90", sizes[len(sizes)*9/10], "max", sizes[len(sizes)-1],
		"avg_communities_per_node", math.Round(float64(memberships)/float64(s.g.numNodes())*100)/100)
}

func (s *mhState) logObjective(msg string, epoch int, stats epochStats) {
	o := s.objective()
	covered := 0.0
	if o.TotalWeight > 0 {
		covered = o.CoveredWeight / o.TotalWeight
	}
	slog.Info(msg, "epoch", epoch, "moved", stats.moved,
		"score", math.Round(o.Score), "covered_weight", math.Round(covered*1e4)/1e4,
		"shared_non_edges", o.SharedNonEdges)
}
