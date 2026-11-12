package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"

	"github.com/urfave/cli/v3"
)

func main() {
	if err := newCommand().Run(context.Background(), os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func newCommand() *cli.Command {
	var profiler *http.Server
	var previousMutexFraction int
	return &cli.Command{
		Name:  "simmy",
		Usage: "SimClusters algorithm implementation",
		Flags: []cli.Flag{
			&cli.StringFlag{Name: "pprof", Usage: "Serve pprof and enable mutex/block profiles at this address (e.g. localhost:9123)"},
		},
		Before: func(ctx context.Context, cmd *cli.Command) (context.Context, error) {
			addr := cmd.String("pprof")
			if addr == "" {
				return ctx, nil
			}
			listener, err := net.Listen("tcp", addr)
			if err != nil {
				return ctx, fmt.Errorf("start pprof: %w", err)
			}
			previousMutexFraction = runtime.SetMutexProfileFraction(1)
			runtime.SetBlockProfileRate(1)
			profiler = &http.Server{Handler: http.DefaultServeMux}
			go func() {
				slog.Info("pprof server listening", "addr", listener.Addr().String())
				if err := profiler.Serve(listener); err != nil && err != http.ErrServerClosed {
					slog.Error("pprof server failed", "error", err)
				}
			}()
			return ctx, nil
		},
		After: func(ctx context.Context, cmd *cli.Command) error {
			if profiler == nil {
				return nil
			}
			runtime.SetMutexProfileFraction(previousMutexFraction)
			runtime.SetBlockProfileRate(0)
			return profiler.Close()
		},
		Commands: []*cli.Command{runCmd, queryCmd, debugUsersCmd, exportCSVCmd, clusterPostsCmd, buildDatasetCmd},
	}
}
