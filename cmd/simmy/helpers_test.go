package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/urfave/cli/v3"
)

func TestAssignmentsFlags(t *testing.T) {
	for _, tt := range []struct {
		name, alias, path string
		flag              cli.Flag
	}{
		{name: "mapping", alias: "m", path: "mapping.json", flag: mappingFlag()},
		{name: "influencers", alias: "i", path: "influencer_communities.json", flag: influencersFlag()},
	} {
		t.Run(tt.name, func(t *testing.T) {
			flag, ok := tt.flag.(*cli.StringFlag)
			if !ok {
				t.Fatalf("flag type = %T, want *cli.StringFlag", tt.flag)
			}
			if flag.Name != tt.name || flag.Value != tt.path || !reflect.DeepEqual(flag.Aliases, []string{tt.alias}) {
				t.Errorf("flag = %+v, want name %q, path %q, alias %q", flag, tt.name, tt.path, tt.alias)
			}
		})
	}
}

func TestLoadAssignments(t *testing.T) {
	for _, tt := range []struct {
		name    string
		json    string
		missing bool
		want    assignments
		wantErr bool
	}{
		{name: "scores", json: `{"42":{"3":0.75,"9":0.25},"7":{}}`, want: assignments{42: {3: 0.75, 9: 0.25}, 7: {}}},
		{name: "empty", json: `{}`, want: assignments{}},
		{name: "int64 IDs", json: `{"9223372036854775807":{"9223372036854775806":1}}`, want: assignments{9223372036854775807: {9223372036854775806: 1}}},
		{name: "invalid JSON", json: `{"42":`, wantErr: true},
		{name: "invalid user ID", json: `{"did:plc:abc":{"3":1}}`, wantErr: true},
		{name: "invalid community ID", json: `{"42":{"abc":1}}`, wantErr: true},
		{name: "invalid score", json: `{"42":{"3":"high"}}`, wantErr: true},
		{name: "missing file", missing: true, wantErr: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "assignments.json")
			if !tt.missing {
				if err := os.WriteFile(path, []byte(tt.json), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			got, err := loadAssignments(path)
			if (err != nil) != tt.wantErr {
				t.Fatalf("loadAssignments() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil {
				if !strings.Contains(err.Error(), path) {
					t.Errorf("error does not identify input path: %v", err)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("loadAssignments() = %v, want %v", got, tt.want)
			}
		})
	}
}
