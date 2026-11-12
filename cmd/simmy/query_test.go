package main

import (
	"reflect"
	"testing"
)

func TestRankInfluencers(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   assignments
		want map[int64][]scoredInfluencer
	}{
		{name: "empty", want: map[int64][]scoredInfluencer{}},
		{
			name: "descending score not ID",
			in:   assignments{1: {5: 0.1}, 99: {5: 0.9}, 3: {5: 0.5}},
			want: map[int64][]scoredInfluencer{5: {{ID: 99, Score: 0.9}, {ID: 3, Score: 0.5}, {ID: 1, Score: 0.1}}},
		},
		{
			name: "ties by ID and multiple communities",
			in:   assignments{9: {5: 0.5, 6: 0.8}, 2: {5: 0.5, 6: 0.2}, 4: {}},
			want: map[int64][]scoredInfluencer{
				5: {{ID: 2, Score: 0.5}, {ID: 9, Score: 0.5}},
				6: {{ID: 9, Score: 0.8}, {ID: 2, Score: 0.2}},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := rankInfluencers(tt.in); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rankInfluencers() = %v, want %v", got, tt.want)
			}
		})
	}
}
