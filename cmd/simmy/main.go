package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	_ "net/http/pprof"
	"os"
	"strconv"
	"sync"

	"github.com/schollz/progressbar/v3"
	"github.com/whyrusleeping/simclusters"
)

func main() {
	if err := runMain(); err != nil {
		fmt.Println(err)
	}
}

func loadEdgesFromFile(simc *simclusters.SimClusters, fn string) error {
	fi, err := os.Open(fn)
	if err != nil {
		return err
	}

	rr := csv.NewReader(fi)

	var edges []simclusters.Edge
	for i := 0; true; i++ {
		rec, err := rr.Read()
		if errors.Is(err, io.EOF) {
			break
		}

		from, err := strconv.Atoi(rec[1])
		if err != nil {
			return err
		}

		to, err := strconv.Atoi(rec[2])
		if err != nil {
			return err
		}

		edges = append(edges, simclusters.Edge{
			From: int64(from),
			To:   int64(to),
		})
	}

	simc.IngestEdges(edges)

	return nil
}

func runMain() error {
	//fi, err := os.Open("/home/why/follows.csv")
	go func() {
		http.ListenAndServe(":9123", nil)
	}()

	cfg := simclusters.DefaultConfig()
	simc := simclusters.NewSimClusters(cfg)

	files := os.Args[1:]

	bar := progressbar.Default(int64(len(files)))
	var wg sync.WaitGroup
	sema := make(chan struct{}, 30)
	for _, fn := range files {
		wg.Add(1)
		sema <- struct{}{}
		go func(file string) {
			defer func() {
				<-sema
				wg.Done()
				bar.Add(1)
			}()

			if err := loadEdgesFromFile(simc, file); err != nil {
				slog.Error("failed to load edges", "file", file, "error", err)
			}
		}(fn)
	}
	wg.Wait()

	if err := simc.ComputeClusters(); err != nil {
		return err
	}

	return nil
}
