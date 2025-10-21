package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"os"
	"strconv"
	"sync"

	"github.com/whyrusleeping/simclusters"
)

func main() {
	if err := runMain(); err != nil {
		fmt.Println(err)
	}
}

func runMain() error {
	//fi, err := os.Open("/home/why/follows.csv")
	fi, err := os.Open("follows2.csv")
	if err != nil {
		return err
	}

	go func() {
		http.ListenAndServe(":9123", nil)
	}()

	recs, err := csv.NewReader(fi).ReadAll()
	if err != nil {
		return err
	}

	cfg := simclusters.DefaultConfig()
	simc := simclusters.NewSimClusters(cfg)

	var edges []simclusters.Edge
	for i, rec := range recs[1:] {
		if i%500 == 0 {
			fmt.Printf("%d / %d    \r", i, len(recs))
		}
		from, err := strconv.Atoi(rec[4])
		if err != nil {
			return err
		}

		to, err := strconv.Atoi(rec[5])
		if err != nil {
			return err
		}

		edges = append(edges, simclusters.Edge{
			From: int64(from),
			To:   int64(to),
		})
	}

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		start := i * len(edges) / 10
		end := (i + 1) * len(edges) / 10
		slice := edges[start:end]
		wg.Add(1)
		go func(subset []simclusters.Edge) {
			defer wg.Done()
			simc.IngestEdges(subset)
		}(slice)
	}
	wg.Wait()

	if err := simc.ComputeClusters(); err != nil {
		return err
	}

	outfi, err := os.Create("output.json")
	if err != nil {
		return err
	}
	defer outfi.Close()

	if err := json.NewEncoder(outfi).Encode(simc.GetClusterIndex().Clusters); err != nil {
		return err
	}

	return nil
}
