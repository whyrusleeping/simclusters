package simclusters

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestInterestedIn(t *testing.T) {
	knownFor := map[int64]map[int]float64{
		1: {10: 0.6, 11: 0.2},
		2: {10: 0.2},
		3: {12: 0.5, 13: 0.5},
	}
	gi := NewGraphIndex()
	gi.AddEdgesBatch([]Edge{
		{From: 1001, To: 1}, {From: 1001, To: 2}, {From: 1001, To: 99}, // 99 is not an influencer
		{From: 1002, To: 1}, {From: 1002, To: 2}, {From: 1002, To: 3},
		{From: 1003, To: 98}, {From: 1003, To: 99},
	})
	x := newInterestIndex(gi, knownFor, 2)
	acc := make([]float64, x.numCommunities)
	user := func(id int64) uint32 {
		u, ok := gi.index(id)
		if !ok {
			t.Fatalf("user %d missing", id)
		}
		return u
	}

	got := x.interestedIn(user(1001), 5, acc, nil)
	want := []communityScore{{10, 0.8}, {11, 0.2}}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i].Community != want[i].Community || math.Abs(got[i].Score-want[i].Score) > 1e-12 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
	for c, v := range acc {
		if v != 0 {
			t.Fatalf("scratch was not cleared: acc[%d] = %v", c, v)
		}
	}

	// Scores are normalized before truncation, and ties break by community.
	got = x.interestedIn(user(1002), 3, acc, got)
	want = []communityScore{{10, 0.4}, {12, 0.25}, {13, 0.25}}
	for i := range want {
		if len(got) != len(want) || got[i].Community != want[i].Community || math.Abs(got[i].Score-want[i].Score) > 1e-12 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}

	if got := x.interestedIn(user(1003), 5, acc, nil); len(got) != 0 {
		t.Errorf("user following no influencer got interests %v", got)
	}
}

func TestWriteInterestedIn(t *testing.T) {
	knownFor := map[int64]map[int]float64{
		1: {10: 0.75, 11: 0.25},
		2: {12: 1},
	}
	gi := NewGraphIndex()
	var edges []Edge
	// Enough users to span several output chunks.
	const users = 10_000
	for u := int64(100); u < 100+users; u++ {
		switch u % 4 {
		case 0:
			edges = append(edges, Edge{From: u, To: 1})
		case 1:
			edges = append(edges, Edge{From: u, To: 2})
		case 2:
			edges = append(edges, Edge{From: u, To: 1}, Edge{From: u, To: 2})
		case 3:
			edges = append(edges, Edge{From: u, To: 50}) // not an influencer
		}
	}
	gi.AddEdgesBatch(edges)

	var first []byte
	for _, workers := range []int{1, 3} {
		var buf bytes.Buffer
		n, err := writeInterestedIn(&buf, gi, knownFor, 5, workers)
		if err != nil {
			t.Fatal(err)
		}
		if want := users * 3 / 4; n != want {
			t.Errorf("wrote %d users, want %d", n, want)
		}
		if first == nil {
			first = buf.Bytes()
		} else if !bytes.Equal(first, buf.Bytes()) {
			t.Error("output depends on the number of workers")
		}

		var got map[int64]map[int]float64
		if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
			t.Fatalf("output is not valid JSON: %v", err)
		}
		if len(got) != n {
			t.Fatalf("decoded %d users, want %d", len(got), n)
		}
		for u, want := range map[int64]map[int]float64{
			100: {10: 0.75, 11: 0.25},
			101: {12: 1},
			102: {10: 0.375, 11: 0.125, 12: 0.5},
		} {
			if !reflect.DeepEqual(got[u], want) {
				t.Errorf("user %d: got %v, want %v", u, got[u], want)
			}
		}
		if _, ok := got[103]; ok {
			t.Error("user following no influencer was written")
		}
	}

	var buf bytes.Buffer
	if n, err := writeInterestedIn(&buf, NewGraphIndex(), knownFor, 5, 2); err != nil || n != 0 || buf.String() != "{}\n" {
		t.Errorf("empty graph: got %q, %d, %v", buf.String(), n, err)
	}
}
