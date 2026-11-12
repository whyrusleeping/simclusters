package main

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/urfave/cli/v3"
	"github.com/whyrusleeping/simclusters"
)

var runCmd = func() *cli.Command {
	d := simclusters.DefaultOptions()
	return &cli.Command{
		Name:      "run",
		Usage:     "Cluster a follow graph",
		ArgsUsage: "follows.csv [more.csv ...]",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:    "checkpoint",
				Aliases: []string{"c"},
				Usage:   "Directory to save the follow graph and similarity graph to, for --from-checkpoint",
			},
			&cli.StringFlag{
				Name:    "from-checkpoint",
				Aliases: []string{"f"},
				Usage:   "Resume from a checkpoint directory instead of reading CSV files",
			},
			&cli.StringFlag{
				Name:    "out-dir",
				Aliases: []string{"o"},
				Usage:   "Directory for " + simclusters.InfluencerCommunitiesFile + " and " + simclusters.MappingFile,
				Value:   d.OutputDir,
			},
			&cli.IntFlag{Name: "min-followers", Usage: "Followers needed to count as an influencer", Value: d.MinFollowers},
			&cli.FloatFlag{Name: "min-similarity", Usage: "Lowest influencer similarity used as an edge when clustering", Value: d.MinSimilarity},
			&cli.FloatFlag{Name: "similarity-floor", Usage: "Lowest influencer similarity stored in the checkpoint, so a resumed run can lower --min-similarity down to it (0 uses --min-similarity)"},
			&cli.IntFlag{Name: "max-neighbors", Usage: "Keep only each influencer's N strongest similarity edges when clustering (0 keeps all)", Value: d.MaxNeighbors},
			&cli.IntFlag{Name: "communities", Usage: "Number of communities", Value: d.NumCommunities},
			&cli.IntFlag{Name: "max-communities", Usage: "Most communities one influencer can belong to (1-7)", Value: d.MaxCommunitiesPerNode},
			&cli.IntFlag{Name: "candidates", Usage: "Communities weighed per influencer per step (up to 8)", Value: d.MaxCandidates},
			&cli.FloatFlag{Name: "alpha", Usage: "Weight of similar pairs sharing a community against dissimilar pairs sharing one", Value: d.Alpha},
			&cli.FloatFlag{Name: "epsilon", Usage: "Chance per step of also weighing a random community", Value: d.Epsilon},
			&cli.FloatFlag{Name: "temperature", Usage: "Sampling temperature; lower is greedier", Value: d.Temperature},
			&cli.IntFlag{Name: "epochs", Usage: "Sampling passes over all influencers", Value: d.Epochs},
			&cli.IntFlag{Name: "greedy-epochs", Usage: "Deterministic improvement passes after sampling", Value: d.GreedyEpochs},
			&cli.IntFlag{Name: "top-interests", Usage: "Communities kept per user", Value: d.TopInterests},
			&cli.Uint64Flag{Name: "seed", Usage: "Random seed; the same seed reproduces the same result", Value: d.Seed},
			&cli.IntFlag{Name: "workers", Usage: "Parallelism", Value: d.Workers},
		},
		Action: runClustering,
	}
}()

func runClustering(ctx context.Context, cmd *cli.Command) error {
	opts := simclusters.DefaultOptions()
	opts.CheckpointPath = cmd.String("checkpoint")
	opts.OutputDir = cmd.String("out-dir")
	opts.MinFollowers = cmd.Int("min-followers")
	opts.SimilarityFloor = cmd.Float("similarity-floor")
	opts.MinSimilarity = cmd.Float("min-similarity")
	opts.MaxNeighbors = cmd.Int("max-neighbors")
	opts.NumCommunities = cmd.Int("communities")
	opts.MaxCommunitiesPerNode = cmd.Int("max-communities")
	opts.MaxCandidates = cmd.Int("candidates")
	opts.Alpha = cmd.Float("alpha")
	opts.Epsilon = cmd.Float("epsilon")
	opts.Temperature = cmd.Float("temperature")
	opts.Epochs = cmd.Int("epochs")
	opts.GreedyEpochs = cmd.Int("greedy-epochs")
	opts.TopInterests = cmd.Int("top-interests")
	opts.Seed = cmd.Uint64("seed")
	opts.Workers = cmd.Int("workers")
	opts.Progress = true

	simc, err := simclusters.New(opts)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(opts.OutputDir, 0o755); err != nil {
		return err
	}

	files := cmd.Args().Slice()
	if from := cmd.String("from-checkpoint"); from != "" {
		if len(files) > 0 {
			return errors.New("--from-checkpoint cannot be combined with input files")
		}
		return simc.ComputeClustersFromCheckpoint(from)
	}
	if len(files) == 0 {
		return errors.New("no input files provided")
	}

	var total int64
	for _, fn := range files {
		st, err := os.Stat(fn)
		if err != nil {
			return err
		}
		total += st.Size()
	}
	start := time.Now()
	bar := progressbar.DefaultBytes(total, "loading follows")
	edges, err := simclusters.LoadEdgesCSV(simc.Graph(), files, simclusters.IngestOptions{
		Workers:  opts.Workers,
		Progress: func(n int64) { bar.Add64(n) },
	})
	bar.Finish()
	if err != nil {
		return err
	}
	slog.Info("follow graph loaded", "edges", edges, "files", len(files), "took", time.Since(start).Round(time.Millisecond))

	return simc.ComputeClusters()
}
