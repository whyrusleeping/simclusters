package simclusters

import (
	"sync"
	"sync/atomic"
)

// parallelRange splits [0, n) into chunks of at most grain items and runs fn
// on them from up to workers goroutines. Chunks are handed out dynamically, so
// uneven per-item cost balances out. worker is a stable index in [0, workers)
// that fn can use to pick per-goroutine scratch space.
func parallelRange(n, workers, grain int, fn func(worker, lo, hi int)) {
	if n <= 0 {
		return
	}
	if grain < 1 {
		grain = 1
	}
	if chunks := (n + grain - 1) / grain; workers > chunks {
		workers = chunks
	}
	if workers <= 1 {
		fn(0, 0, n)
		return
	}

	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for {
				lo := int(next.Add(int64(grain))) - grain
				if lo >= n {
					return
				}
				fn(w, lo, min(lo+grain, n))
			}
		}(w)
	}
	wg.Wait()
}

// splitmix64 is the SplitMix64 finalizer, used to derive well-mixed seeds and
// hash values from sequential inputs.
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}
