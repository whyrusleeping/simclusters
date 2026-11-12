package main

import (
	"context"
	"io"
	"runtime"
	"testing"

	"github.com/urfave/cli/v3"
)

func TestProfilingOptIn(t *testing.T) {
	previous := runtime.SetMutexProfileFraction(0)
	defer runtime.SetMutexProfileFraction(previous)

	for _, tt := range []struct {
		name    string
		flags   []string
		profile int
		wantErr bool
	}{
		{name: "off by default"},
		{name: "opt in", flags: []string{"--pprof", "localhost:0"}, profile: 1},
		{name: "invalid address", flags: []string{"--pprof", "invalid address"}, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			app := newCommand()
			app.Writer, app.ErrWriter = io.Discard, io.Discard
			called := false
			app.Commands = []*cli.Command{{Name: "check", Action: func(ctx context.Context, cmd *cli.Command) error {
				called = true
				if got := runtime.SetMutexProfileFraction(-1); got != tt.profile {
					t.Errorf("mutex profiling fraction = %d, want %d", got, tt.profile)
				}
				return nil
			}}}
			args := append([]string{"simmy"}, tt.flags...)
			err := app.Run(context.Background(), append(args, "check"))
			if (err != nil) != tt.wantErr {
				t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
			}
			if called == tt.wantErr {
				t.Errorf("action called = %v, want %v", called, !tt.wantErr)
			}
			if got := runtime.SetMutexProfileFraction(-1); got != 0 {
				t.Errorf("mutex profiling not restored: %d", got)
			}
		})
	}
}
