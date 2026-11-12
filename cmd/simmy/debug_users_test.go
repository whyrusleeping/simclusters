package main

import (
	"math"
	"slices"
	"testing"

	"github.com/whyrusleeping/simclusters"
)

func TestCollectUserEdgeStats(t *testing.T) {
	for _, tt := range []struct {
		name  string
		users []int64
		edges []simclusters.SimEdge
		want  map[int64]userEdgeStats
	}{
		{name: "no users", edges: []simclusters.SimEdge{{A: 1, B: 2, Weight: 0.2}}, want: map[int64]userEdgeStats{}},
		{name: "no edges", users: []int64{1}, want: map[int64]userEdgeStats{1: {}}},
		{
			name:  "both endpoints and exact thresholds",
			users: []int64{1, 2, 3, 4},
			edges: []simclusters.SimEdge{
				{A: 1, B: 2, Weight: 0.1},
				{A: 2, B: 3, Weight: 0.15},
				{A: 3, B: 1, Weight: 0.2},
				{A: 1, B: 99, Weight: 0.05},
				{A: 98, B: 99, Weight: 0.9},
			},
			want: map[int64]userEdgeStats{
				1: {Count: 3, Above01: 2, Above015: 1, Above02: 1, Min: 0.05, Max: 0.2, Sum: 0.35},
				2: {Count: 2, Above01: 2, Above015: 1, Min: 0.1, Max: 0.15, Sum: 0.25},
				3: {Count: 2, Above01: 2, Above015: 2, Above02: 1, Min: 0.15, Max: 0.2, Sum: 0.35},
				4: {},
			},
		},
		{
			name:  "duplicate users and self edge counted once",
			users: []int64{1, 1},
			edges: []simclusters.SimEdge{{A: 1, B: 1, Weight: 0.5}},
			want:  map[int64]userEdgeStats{1: {Count: 1, Above01: 1, Above015: 1, Above02: 1, Min: 0.5, Max: 0.5, Sum: 0.5}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := collectUserEdgeStats(tt.users, slices.Values(tt.edges))
			if len(got) != len(tt.want) {
				t.Fatalf("got %d users, want %d", len(got), len(tt.want))
			}
			for id, want := range tt.want {
				s, ok := got[id]
				if !ok || s.Count != want.Count || s.Above01 != want.Above01 || s.Above015 != want.Above015 || s.Above02 != want.Above02 || s.Min != want.Min || s.Max != want.Max || math.Abs(s.Sum-want.Sum) > 1e-12 {
					t.Errorf("stats[%d] = %+v (present: %v), want %+v", id, s, ok, want)
				}
			}
		})
	}
}
