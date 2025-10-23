package main

import (
	"context"
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
	"github.com/urfave/cli/v3"
	"github.com/whyrusleeping/simclusters"
)

func main() {
	// Start pprof server
	go func() {
		http.ListenAndServe(":9123", nil)
	}()

	cmd := &cli.Command{
		Name:  "simmy",
		Usage: "SimClusters algorithm implementation",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "checkpoint",
				Aliases: []string{"c"},
				Usage:   "Path to save checkpoint after initializeInfluencerEdges",
			},
			&cli.StringFlag{
				Name:    "from-checkpoint",
				Aliases: []string{"f"},
				Usage:   "Path to load checkpoint and resume computation",
			},
		},
		Action: runClustering,
	}

	if err := cmd.Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func loadEdgesFromFile(simc *simclusters.SimClusters, fn string) error {
	fi, err := os.Open(fn)
	if err != nil {
		return err
	}
	defer fi.Close()

	rr := csv.NewReader(fi)

	var edges []simclusters.Edge
	for i := 0; true; i++ {
		rec, err := rr.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
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

func runClustering(ctx context.Context, cmd *cli.Command) error {
	cfg := simclusters.DefaultConfig()
	simc := simclusters.NewSimClusters(cfg)

	checkpointPath := cmd.String("checkpoint")
	fromCheckpoint := cmd.String("from-checkpoint")

	// Set checkpoint path if provided
	if checkpointPath != "" {
		simc.CheckpointPath = checkpointPath
		slog.Info("Checkpoint will be saved to", "path", checkpointPath)
	}

	// If resuming from checkpoint, skip data loading
	if fromCheckpoint != "" {
		slog.Info("Resuming computation from checkpoint", "path", fromCheckpoint)
		return simc.ComputeClustersFromCheckpoint(fromCheckpoint)
	}

	// Load data files
	files := cmd.Args().Slice()
	if len(files) == 0 {
		return fmt.Errorf("no input files provided")
	}

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
